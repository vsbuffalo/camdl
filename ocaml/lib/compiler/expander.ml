(* Expander: AST declarations → Ir.model *)

open Ast

(* ── Context ─────────────────────────────────────────────────────────────── *)

type context = {
  mutable time_unit       : unit_lit;
  mutable description     : string option;
  mutable comp_decls      : compartment_decl list;
  mutable param_decls     : param_decl list;
  mutable let_bindings    : let_binding list;
  mutable stratifies      : stratify_decl list;
  mutable transitions     : transition_decl list;  (* post-desugar *)
  mutable orig_transitions: transition_decl list;  (* pre-desugar original *)
  mutable init_entries    : init_entry list;
  mutable simulate        : simulate_decl option;
  mutable ode_decls       : ode_decl list;
  mutable func_decls      : func_decl list;
  mutable obs_decls       : obs_decl list;
  mutable interv_decls    : intervention_decl list;
  mutable output_decl     : output_decl option;
  mutable table_decls     : table_decl list;
  mutable scenario_decls  : scenario_decl list;
  mutable balance_decl    : balance_decl option;
  mutable event_decls     : intervention_decl list;
  mutable diags           : Diagnostics.t;  (* collected errors/warnings *)
  mutable source_dir      : string;         (* directory of the source file *)
  mutable filename        : string;         (* source filename for diagnostic locs *)
  mutable expanded_comp_cache : string list;
  mutable dim_decls       : dimensions_entry list;
  mutable dim_registry    : (string * string list) list;
  (* dim name → ordered levels; populated by resolve_dimensions pass *)
  mutable origin          : string option;
  (* ISO date string for date() → float conversion *)
  (* O(1) lookup tables — populated by build_lookup_tables after resolve_dimensions *)
  mutable let_tbl         : (string, let_binding) Hashtbl.t;
  mutable comp_tbl        : (string, compartment_decl) Hashtbl.t;
  mutable scalar_param_tbl: (string, unit) Hashtbl.t;
  mutable expanded_param_tbl : (string, unit) Hashtbl.t;
  mutable func_tbl        : (string, func_decl) Hashtbl.t;
  mutable expanded_comp_tbl  : (string, unit) Hashtbl.t;
}

let empty_context ?(source_dir = "") ?(filename = "<input>") () = {
  time_unit        = Days;
  description      = None;
  comp_decls       = [];
  param_decls      = [];
  let_bindings     = [];
  stratifies       = [];
  transitions      = [];
  orig_transitions = [];
  init_entries     = [];
  simulate         = None;
  ode_decls        = [];
  func_decls       = [];
  obs_decls        = [];
  interv_decls     = [];
  output_decl      = None;
  table_decls          = [];
  scenario_decls       = [];
  balance_decl         = None;
  event_decls          = [];
  diags                = Diagnostics.create ();
  source_dir;
  filename;
  expanded_comp_cache  = [];
  dim_decls            = [];
  dim_registry         = [];
  origin               = None;
  let_tbl              = Hashtbl.create 16;
  comp_tbl             = Hashtbl.create 16;
  scalar_param_tbl     = Hashtbl.create 16;
  expanded_param_tbl   = Hashtbl.create 16;
  func_tbl             = Hashtbl.create 16;
  expanded_comp_tbl    = Hashtbl.create 16;
}

(* ── Model summary ────────────────────────────────────────────────────────── *)

type model_summary = {
  base_compartment_count    : int;
  expanded_compartment_count: int;
  base_transition_count     : int;
  expanded_transition_count : int;
  filtered_transition_count : int;
  let_binding_count         : int;
  table_count               : int;
  param_count               : int;
  obs_count                 : int;
  interv_count              : int;
}

(* ── Date arithmetic ─────────────────────────────────────────────────────── *)

(** Proleptic Gregorian day number (relative to an internal epoch).
    Formula from Hatcher / Richards — works for dates CE 1583+. *)
let days_of_date y m d =
  let y' = if m <= 2 then y - 1 else y in
  let m' = if m <= 2 then m + 12 else m in
  365 * y' + y'/4 - y'/100 + y'/400 + (153*(m'+1))/5 + d - 694025

let parse_iso_date s =
  match String.split_on_char '-' s with
  | [ys; ms; ds] ->
    (try (int_of_string ys, int_of_string ms, int_of_string ds)
     with _ -> failwith (Printf.sprintf "invalid date literal '%s': components must be integers" s))
  | _ -> failwith (Printf.sprintf "date literal must be YYYY-MM-DD, got '%s'" s)

let parse_date_to_float origin_str date_str time_unit =
  let (oy, om, od) = parse_iso_date origin_str in
  let (ty, tm, td) = parse_iso_date date_str in
  let delta = days_of_date ty tm td - days_of_date oy om od in
  (* days_per is defined below; forward-declare not needed since
     parse_date_to_float is only called after full initialization.
     Use the same Gregorian constant (365.2425) everywhere. *)
  let days = function
    | Days | PerDay -> 1.0
    | Weeks | PerWeek -> 7.0
    | Months | PerMonth -> 365.2425 /. 12.0
    | Years | PerYear -> 365.2425
    | Count | Ratio ->
      (* Unreachable: time_unit is validated to be a time unit at
         parse time. Non-time unit here means upstream malformed the AST. *)
      invalid_arg "parse_date_to_float: time_unit must be a time unit"
  in
  float_of_int delta /. days time_unit

(* ── Data loading helpers ─────────────────────────────────────────────────── *)

(** Resolve a path relative to source_dir.  Absolute paths pass through. *)
let resolve_data_path ctx path =
  if Filename.is_relative path && ctx.source_dir <> "" then
    Filename.concat ctx.source_dir path
  else path

(** Split a line by a separator character, returning a list of fields. *)
let split_by sep line =
  let parts = ref [] in
  let buf   = Buffer.create 16 in
  String.iter (fun c ->
    if c = sep then (parts := Buffer.contents buf :: !parts; Buffer.clear buf)
    else Buffer.add_char buf c
  ) line;
  parts := Buffer.contents buf :: !parts;
  List.rev !parts

(** Read a CSV/TSV file, calling [on_header] with the header fields and
    [on_row] with each data row's fields (trimmed, non-empty, non-comment lines).
    Handles path resolution, extension-based separator detection, and error
    reporting. Returns [None] if the file is missing; [Some result] from
    [on_done] otherwise. [on_done] is called after all rows, before close. *)
let read_csv_rows ctx path ~on_header ~on_row ~on_done =
  let abs_path = resolve_data_path ctx path in
  if not (Sys.file_exists abs_path) then begin
    Diagnostics.error ctx.diags
      ~code:"E200"
      ~loc:Diagnostics.no_loc
      ~message:(Printf.sprintf "data file not found: %s" path)
      ~hint:"check the path is relative to the .camdl source file"
      ();
    None
  end else begin
    let ext = String.lowercase_ascii (Filename.extension path) in
    let sep = match ext with
      | ".csv" -> ','
      | ".tsv" -> '\t'
      | _ ->
        Diagnostics.error ctx.diags
          ~code:"E205"
          ~loc:Diagnostics.no_loc
          ~message:(Printf.sprintf "unrecognized extension '%s' in %s; use .csv or .tsv" ext path)
          ();
        '\t'
    in
    let ic = open_in abs_path in
    (* M8 in the 2026-04-19 review: previously this sequence was
       `let result = try ... in close_in ic; Some result` — any
       non-End_of_file exception (I/O errors, failed assertions
       inside a callback, etc.) propagated past the try-block and
       close_in was never reached, leaking the file descriptor.
       Fun.protect guarantees the close runs on any exit path,
       normal or exceptional. *)
    let result = Fun.protect ~finally:(fun () -> close_in_noerr ic) (fun () ->
      try
        let header_line = input_line ic in
        let header_cols = List.map String.trim (split_by sep header_line) in
        on_header header_cols;
        let row_num = ref 1 in
        (try while true do
          let raw_line = input_line ic in
          incr row_num;
          let line = String.trim raw_line in
          if line <> "" && not (String.length line > 0 && line.[0] = '#') then begin
            let cols = split_by sep line in
            on_row !row_num cols
          end
        done with End_of_file -> ());
        on_done ()
      with End_of_file ->
        Diagnostics.error ctx.diags
          ~code:"E210"
          ~loc:Diagnostics.no_loc
          ~message:(Printf.sprintf "%s: file is empty (no header row)" path)
          ();
        on_done ()
    ) in
    Some result
  end

(** Load a `read(path, ...)` file → list of n_values float arrays (row-major).
    The file must have a header row.
    dims is the list of table_dim_entry (TDim/TDimUnit) for index columns.
    n_values is the number of value columns (= List.length tnames).
    default_val = Some f → sparse (missing cells get f); None → dense (all cells required). *)
let load_table_data ctx path ~dims ~n_values ~default_val =
  let n_dims = List.length dims in
  (* Compute dimension sizes and level lists *)
  let dim_info = List.map (fun de ->
    let dname = match de with
      | TDim d | TDimUnit (d, _) -> d
    in
    let levels = match List.assoc_opt dname ctx.dim_registry with
      | Some vs -> vs
      | None    -> []
    in
    (dname, levels)
  ) dims in
  let dim_sizes = List.map (fun (_, lvs) -> List.length lvs) dim_info in
  let total = List.fold_left ( * ) 1 dim_sizes in
  (* Allocate arrays; use nan as sentinel for dense-check *)
  let sentinel = match default_val with
    | Some f -> f
    | None   -> Float.nan
  in
  let arrays = Array.init n_values (fun _ -> Array.make total sentinel) in
  (* Keep track of which cells were set, for duplicate detection *)
  let set_flags = Array.make total false in
  (* Precompute strides: dim 0 has stride = product of all later dim sizes *)
  let strides = Array.make n_dims 1 in
  for i = n_dims - 2 downto 0 do
    strides.(i) <- strides.(i + 1) * (List.nth dim_sizes (i + 1))
  done;
  let dim_names = List.map fst dim_info in
  let on_header header_cols =
    let header_dims =
      List.init (min n_dims (List.length header_cols))
        (fun i -> String.trim (List.nth header_cols i))
    in
    if header_dims <> dim_names then begin
      let header_sorted = List.sort compare header_dims in
      let expected_sorted = List.sort compare dim_names in
      if header_sorted = expected_sorted then
        Diagnostics.error ctx.diags
          ~code:"E216"
          ~loc:Diagnostics.no_loc
          ~message:(Printf.sprintf
            "%s: dimension columns appear reordered; expected %s, got %s"
            path
            (String.concat ", " dim_names)
            (String.concat ", " header_dims))
          ()
      else
        List.iteri (fun i (expected, actual) ->
          if expected <> actual then
            Diagnostics.warning ctx.diags
              ~code:"W201"
              ~loc:Diagnostics.no_loc
              ~message:(Printf.sprintf
                "%s: column %d is named '%s' but maps to dimension '%s'"
                path (i + 1) actual expected)
              ()
        ) (List.combine dim_names header_dims)
    end
  in
  let on_row row_num cols =
    let ncols = List.length cols in
    let expected = n_dims + n_values in
    if ncols <> expected then begin
      Diagnostics.error ctx.diags
        ~code:"E206"
        ~loc:Diagnostics.no_loc
        ~message:(Printf.sprintf "%s row %d: expected %d columns (%d dim + %d value), got %d"
          path row_num expected n_dims n_values ncols)
        ()
    end else begin
      (* Compute flat index from dim columns *)
      let flat_idx = ref 0 in
      let ok = ref true in
      List.iteri (fun i de ->
        let dname, levels = List.nth dim_info i in
        let cell = String.trim (List.nth cols i) in
        (match List.find_index (fun v -> v = cell) levels with
         | Some idx ->
           flat_idx := !flat_idx + idx * strides.(i)
         | None ->
           Diagnostics.error ctx.diags
             ~code:"E207"
             ~loc:Diagnostics.no_loc
             ~message:(Printf.sprintf "'%s' in column %d of %s is not a valid '%s' level"
               cell (i + 1) path dname)
             ();
           ok := false);
        ignore de
      ) dims;
      if !ok then begin
        let idx = !flat_idx in
        if set_flags.(idx) then begin
          Diagnostics.error ctx.diags
            ~code:"E208"
            ~loc:Diagnostics.no_loc
            ~message:(Printf.sprintf "%s row %d: duplicate key" path row_num)
            ()
        end else begin
          set_flags.(idx) <- true;
          for j = 0 to n_values - 1 do
            let cell = String.trim (List.nth cols (n_dims + j)) in
            match float_of_string_opt cell with
            | Some f -> arrays.(j).(idx) <- f
            | None ->
              Diagnostics.error ctx.diags
                ~code:"E209"
                ~loc:Diagnostics.no_loc
                ~message:(Printf.sprintf "%s row %d column %d: expected a number, got '%s'"
                  path row_num (n_dims + j + 1) cell)
                ()
          done
        end
      end
    end
  in
  let on_done () =
    (* Dense check: if no default_val, all cells must have been set *)
    if default_val = None then begin
      for idx = 0 to total - 1 do
        if not set_flags.(idx) then begin
          (* Find which dim combination this idx corresponds to *)
          let coords = ref [] in
          let rem = ref idx in
          for i = 0 to n_dims - 1 do
            let q = !rem / strides.(i) in
            rem := !rem mod strides.(i);
            let (dname, levels) = List.nth dim_info i in
            let level = if q < List.length levels then List.nth levels q else "?" in
            coords := (dname ^ "=" ^ level) :: !coords
          done;
          let coord_str = String.concat ", " (List.rev !coords) in
          Diagnostics.error ctx.diags
            ~code:"E211"
            ~loc:Diagnostics.no_loc
            ~message:(Printf.sprintf "missing entry for (%s) in %s" coord_str path)
            ()
        end
      done;
      (* M6 in 2026-04-19 review: replace any remaining NaN sentinels
         with 0.0 so that a caller who ignores has_errors can't emit
         NaN values into the IR. Diagnostics are still attached; the
         pipeline will report_and_exit on E211 before the IR is
         serialized. *)
      Array.iter (fun arr ->
        for i = 0 to Array.length arr - 1 do
          if Float.is_nan arr.(i) then arr.(i) <- 0.0
        done
      ) arrays
    end;
    Array.to_list arrays
  in
  match read_csv_rows ctx path ~on_header ~on_row ~on_done with
  | Some result -> result
  | None -> List.init n_values (fun _ -> [||])

(* Convert an Ast.loc into a Diagnostics.loc. If the AST loc's file
   field is empty (parser didn't know the filename), substitute the
   ctx's filename so diagnostics show the correct `file:line:col`
   header. *)
let diag_loc_of_ast_ctx ctx (l : Ast.loc) : Diagnostics.loc =
  let file = if l.file = "" then ctx.filename else l.file in
  { Diagnostics.file; line = l.line; col = l.col;
    end_line = l.end_line; end_col = l.end_col }

let reserved_time_names = ["t"; "t_start"; "t_end"; "dt"]
let reserved_math_names = ["pi"; "e"]                       (* gh#58 *)

let check_reserved ?(loc = Diagnostics.no_loc) ctx name kind =
  if List.mem name reserved_time_names then
    Diagnostics.error ctx.diags ~code:"E100" ~loc
      ~message:(Printf.sprintf "%s name '%s' is reserved for simulation time" kind name)
      ~hint:"choose a different name" ()
  else if List.mem name reserved_math_names then
    Diagnostics.error ctx.diags ~code:"E100" ~loc
      ~message:(Printf.sprintf "%s name '%s' is reserved (math constant)" kind name)
      ~hint:"choose a different name" ()

let collect_declarations ctx decls =
  (* Use List.rev_append (prepend reversed chunk) during iteration, then
     reverse each list once at the end.  This avoids O(n) per append. *)
  List.iter (fun d -> match d with
    | DTimeUnit u        -> ctx.time_unit <- u
    | DDescription s     -> ctx.description <- Some s
    | DOrigin s          -> ctx.origin <- Some s
    | DDimensions es     -> ctx.dim_decls <- List.rev_append es ctx.dim_decls
    | DCompartments cs   ->
      List.iter (fun (c : compartment_decl) ->
        check_reserved ctx ~loc:(diag_loc_of_ast_ctx ctx c.cloc) c.cname "compartment") cs;
      ctx.comp_decls <- List.rev_append cs ctx.comp_decls
    | DParameters ps     ->
      List.iter (fun p -> match p with
        | PScalar s  -> check_reserved ctx ~loc:(diag_loc_of_ast_ctx ctx s.ploc) s.pname "parameter"
        | PIndexed s -> check_reserved ctx ~loc:(diag_loc_of_ast_ctx ctx s.ploc) s.pname "parameter") ps;
      ctx.param_decls <- List.rev_append ps ctx.param_decls
    | DLet lb            ->
      check_reserved ctx lb.lname "let binding";
      ctx.let_bindings <- lb :: ctx.let_bindings
    | DStratify sd       ->
      ctx.stratifies <- sd :: ctx.stratifies
    | DTransitions trs   -> ctx.transitions <- List.rev_append trs ctx.transitions
    | DInit ies          -> ctx.init_entries <- List.rev_append ies ctx.init_entries
    | DSimulate sd       -> ctx.simulate <- Some sd
    | DODE odes          -> ctx.ode_decls <- List.rev_append odes ctx.ode_decls
    | DForcing fs        -> ctx.func_decls <- List.rev_append fs ctx.func_decls
    | DObservations obs  -> ctx.obs_decls <- List.rev_append obs ctx.obs_decls
    | DInterventions ivs -> ctx.interv_decls <- List.rev_append ivs ctx.interv_decls
    | DOutput od         -> ctx.output_decl <- Some od
    | DTables tds        -> ctx.table_decls <- List.rev_append tds ctx.table_decls
    | DTimepoints _      -> ()
    | DScenarios ss      -> ctx.scenario_decls <- List.rev_append ss ctx.scenario_decls
    | DBalance bd        -> ctx.balance_decl <- Some bd
    | DEvents evs        -> ctx.event_decls <- List.rev_append evs ctx.event_decls
  ) decls;
  (* Reverse all accumulated lists to restore declaration order *)
  ctx.dim_decls      <- List.rev ctx.dim_decls;
  ctx.comp_decls     <- List.rev ctx.comp_decls;
  ctx.param_decls    <- List.rev ctx.param_decls;
  ctx.let_bindings   <- List.rev ctx.let_bindings;
  ctx.stratifies     <- List.rev ctx.stratifies;
  ctx.transitions    <- List.rev ctx.transitions;
  ctx.init_entries   <- List.rev ctx.init_entries;
  ctx.ode_decls      <- List.rev ctx.ode_decls;
  ctx.func_decls     <- List.rev ctx.func_decls;
  ctx.obs_decls      <- List.rev ctx.obs_decls;
  ctx.interv_decls   <- List.rev ctx.interv_decls;
  ctx.table_decls    <- List.rev ctx.table_decls;
  ctx.scenario_decls <- List.rev ctx.scenario_decls;
  ctx.event_decls    <- List.rev ctx.event_decls;
  ctx.orig_transitions <- ctx.transitions

(* ── Dimensions pass ─────────────────────────────────────────────────────── *)

(** Read unique values from a named column in a file, preserving first-occurrence order.
    Returns (levels, n_rows, n_duplicates). *)
let read_dim_column_from_file ctx path col_name =
  let col_pos = ref (-1) in
  let seen = Hashtbl.create 16 in
  let order = ref [] in
  let n_rows = ref 0 in
  let n_dups = ref 0 in
  let on_header headers =
    (match List.find_index (fun h -> h = col_name) headers with
     | Some i -> col_pos := i
     | None ->
       Diagnostics.error ctx.diags
         ~code:"E218"
         ~loc:Diagnostics.no_loc
         ~message:(Printf.sprintf "column '%s' not found in %s (headers: %s)"
           col_name path (String.concat ", " headers))
         ())
  in
  let on_row _row_num cols =
    incr n_rows;
    if !col_pos >= 0 then
      match List.nth_opt cols !col_pos with
      | None -> ()
      | Some cell ->
        let v = String.trim cell in
        if v <> "" then begin
          if Hashtbl.mem seen v then incr n_dups
          else begin
            Hashtbl.add seen v ();
            order := v :: !order
          end
        end
  in
  let on_done () = (List.rev !order, !n_rows, !n_dups) in
  match read_csv_rows ctx path ~on_header ~on_row ~on_done with
  | Some result -> result
  | None -> ([], 0, 0)

(** Pass 1: process DDimensions declarations, build dim_registry.
    Emits info messages for file-derived dimensions. *)
let resolve_dimensions ctx =
  List.iter (fun de ->
    if List.mem_assoc de.dename ctx.dim_registry then
      Diagnostics.error ctx.diags
        ~code:"E212"
        ~loc:Diagnostics.no_loc
        ~message:(Printf.sprintf "dimension '%s' is declared more than once in dimensions {}" de.dename)
        ()
    else begin
      let levels = match de.desrc with
        | DInline vs -> vs
        | DRead { fn_name; path; col_kw; col } ->
          (* M11 in 2026-04-19 review: parser accepts any
             `IDENT(STRING, IDENT = STRING)`, so `load("pop.tsv",
             column = "patch")` parses identically to `read(...,
             banana = "patch")`. Validate the function name and
             keyword here with proper diagnostics. *)
          if fn_name <> "read" then
            Diagnostics.error ctx.diags
              ~code:"E275"
              ~loc:Diagnostics.no_loc
              ~message:(Printf.sprintf
                "unknown dimension source function '%s' — use `read(...)`"
                fn_name)
              ~hint:"example: patch = read(\"pop.tsv\", column = \"patch\")"
              ();
          if col_kw <> "column" then
            Diagnostics.error ctx.diags
              ~code:"E276"
              ~loc:Diagnostics.no_loc
              ~message:(Printf.sprintf
                "unknown keyword '%s' for read(...) — use `column = \"...\"`"
                col_kw)
              ~hint:"valid keywords: column"
              ();
          let (vs, n_rows, n_dups) = read_dim_column_from_file ctx path col in
          (* Previously this site printed an "info: dimension '%s': N
             levels from..." line via Printf.eprintf — M7 in the
             2026-04-19 review. That bypassed Diagnostics, couldn't
             be silenced or JSONified, and always fired even in
             `camdlc compile model.camdl > out.json` where the user
             wants only JSON on stdout. Suppressed entirely; the same
             information is surfaced via `camdlc inspect --dims`
             when a user wants it. If the duplicate count is
             informative (n_dups > 0), surface as a warning so it
             rides the proper diagnostics channel. *)
          if n_dups > 0 then
            Diagnostics.warning ctx.diags
              ~code:"W311"
              ~loc:Diagnostics.no_loc
              ~message:(Printf.sprintf
                "dimension '%s' from %s column \"%s\": %d duplicate rows \
                 collapsed to %d unique levels (of %d total)"
                de.dename path col n_dups (List.length vs) n_rows)
              ();
          vs
      in
      ctx.dim_registry <- ctx.dim_registry @ [(de.dename, levels)]
    end
  ) ctx.dim_decls;
  (* Validate: every stratify dimension must be in dim_registry *)
  List.iter (fun sd ->
    if not (List.mem_assoc sd.sdim ctx.dim_registry) then
      Diagnostics.error ctx.diags
        ~code:"E214"
        ~loc:Diagnostics.no_loc
        ~message:(Printf.sprintf
          "stratify(by = '%s') has no levels: declare it in dimensions { %s = [...] }"
          sd.sdim sd.sdim)
        ()
  ) ctx.stratifies

(* ── Unit conversion ─────────────────────────────────────────────────────── *)

(* Number of days represented by each unit literal. Used as the universal
   intermediate: to convert between any two units, go via days. *)
(* Gregorian average year = 365.2425 days. Used as the universal intermediate
   for all unit conversions. Must match parse_date_to_float above. *)
let days_per = function
  | Days     -> 1.0              | PerDay   -> 1.0
  | Weeks    -> 7.0              | PerWeek  -> 7.0
  | Months   -> 365.2425 /. 12.0 | PerMonth -> 365.2425 /. 12.0
  | Years    -> 365.2425          | PerYear  -> 365.2425
  | Count | Ratio ->
    (* `days_per` is the time-scale machinery; non-time units don't
       have a time-per-unit ratio. Callers must dispatch on the unit
       kind and avoid calling this on `'count` / `'ratio`. *)
    invalid_arg "days_per: non-time unit has no time scale"

(* Convert a unit literal expression to a float in the model's declared
   time_unit.  The computation goes through days as the universal intermediate:
     duration:  f 'u  = (f × days_per(u)) / days_per(time_unit)
     rate:      f 'pu = (f / days_per(u)) × days_per(time_unit)

   With time_unit = 'days (the common case) days_per(Days) = 1.0, so the
   division/multiplication is a no-op and the result is identical to the
   old hardcoded behaviour.  With time_unit = 'weeks, 80 'days → 80/7 ≈ 11.4
   and 0.3 'per_day → 0.3 × 7 = 2.1. *)
let unit_lit_to_string = function
  | Days -> "days" | Weeks -> "weeks" | Months -> "months" | Years -> "years"
  | PerDay -> "per_day" | PerWeek -> "per_week" | PerMonth -> "per_month" | PerYear -> "per_year"
  | Count -> "count" | Ratio -> "ratio"

let unit_to_model_time ctx f u =
  let tu = days_per ctx.time_unit in
  match u with
  | Days | Weeks | Months | Years ->
    f *. days_per u /. tu
  | PerDay | PerWeek | PerMonth | PerYear ->
    f /. days_per u *. tu
  | Count | Ratio ->
    (* Tier-3 non-time units (GH #8). No time scale; values pass
       through unchanged. The dimension information is captured via
       `unit_lit_to_dim` and stored on the `time_function.dim` field
       for the dim-checker to consume. *)
    f

(** Dimension tuple (P_exp, T_exp) for a unit literal. Used when a
    unit literal annotates a tier-3 declaration (tables, forcing
    functions — GH #8) to drive dimensional analysis. *)
let unit_lit_to_dim = function
  | Days | Weeks | Months | Years      -> (0, 1)   (* time *)
  | PerDay | PerWeek | PerMonth | PerYear -> (0, -1) (* rate *)
  | Count                              -> (1, 0)   (* population *)
  | Ratio                              -> (0, 0)   (* dimensionless multiplier *)

(* ── Stratification helpers ──────────────────────────────────────────────── *)

let dim_values ctx dim =
  match List.assoc_opt dim ctx.dim_registry with
  | Some vs -> vs
  | None    -> []

let strat_applies_to _ctx cname sd =
  match sd.sonly with
  | None      -> true
  | Some only -> List.mem cname only

let comp_dims ctx cname =
  List.filter_map (fun sd ->
    if strat_applies_to ctx cname sd then Some sd.sdim else None
  ) ctx.stratifies

let expand_compartment_name ctx cname =
  let dims = comp_dims ctx cname in
  if dims = [] then [cname]
  else begin
    let all_vals = List.map (fun d -> (d, dim_values ctx d)) dims in
    let rec cart = function
      | [] -> [[]]
      | (_, vs) :: rest ->
        let tails = cart rest in
        List.concat_map (fun v -> List.map (fun t -> v :: t) tails) vs
    in
    List.map (fun combo -> String.concat "_" (cname :: combo)) (cart all_vals)
  end

let all_expanded_compartments ctx =
  List.concat_map (fun cd -> expand_compartment_name ctx cd.cname) ctx.comp_decls

(** Build O(1) lookup tables from the declaration lists and dim_registry.
    Call after resolve_dimensions so expanded indexed param names are known. *)
let build_lookup_tables ctx =
  (* let bindings: name -> binding *)
  let lt = Hashtbl.create (List.length ctx.let_bindings) in
  List.iter (fun lb -> Hashtbl.replace lt lb.lname lb) ctx.let_bindings;
  ctx.let_tbl <- lt;
  (* compartment decls: name -> decl *)
  let ct = Hashtbl.create (List.length ctx.comp_decls) in
  List.iter (fun cd -> Hashtbl.replace ct cd.cname cd) ctx.comp_decls;
  ctx.comp_tbl <- ct;
  (* scalar params: name -> unit *)
  let spt = Hashtbl.create (List.length ctx.param_decls) in
  List.iter (fun pd -> match pd with
    | PScalar p -> Hashtbl.replace spt p.pname ()
    | _ -> ()
  ) ctx.param_decls;
  ctx.scalar_param_tbl <- spt;
  (* expanded indexed param names: "R0_urban" etc. -> unit *)
  let ept = Hashtbl.create 16 in
  List.iter (fun pd -> match pd with
    | PIndexed { pname; pdims = [dim]; _ } ->
      let vals = match List.assoc_opt dim ctx.dim_registry with
        | Some vs -> vs | None -> []
      in
      List.iter (fun v -> Hashtbl.replace ept (pname ^ "_" ^ v) ()) vals
    | _ -> ()
  ) ctx.param_decls;
  ctx.expanded_param_tbl <- ept;
  (* func decls: name -> decl *)
  let ft = Hashtbl.create (List.length ctx.func_decls) in
  List.iter (fun (fd : func_decl) -> Hashtbl.replace ft fd.fname fd) ctx.func_decls;
  ctx.func_tbl <- ft;
  (* expanded compartment names: prime the hash table and cache *)
  let ec = Hashtbl.create 64 in
  let expanded = all_expanded_compartments ctx in
  List.iter (fun n -> Hashtbl.replace ec n ()) expanded;
  ctx.expanded_comp_tbl <- ec;
  ctx.expanded_comp_cache <- expanded

(* ── Table helpers ───────────────────────────────────────────────────────── *)

let dim_name_of_entry = function
  | TDim d | TDimUnit (d, _) -> d

(** Extract the unit literal from a table's dim list, if any.
    Spec §6.1 permits at most one unit annotation per table (the annotation
    is logically on the value, not on a particular dim); parser grammar
    allows multiple, so we enforce the invariant here. *)
let extract_table_unit ctx ~table_name (dims : table_dim_entry list) =
  let units = List.filter_map (function
    | TDim _ -> None
    | TDimUnit (_, u) -> Some u
  ) dims in
  match units with
  | [] -> None
  | [u] -> Some u
  | _ ->
    Diagnostics.error ctx.diags
      ~code:"E216"
      ~loc:Diagnostics.no_loc
      ~message:(Printf.sprintf
        "table '%s' has unit annotations on more than one dimension; \
         declare the unit on exactly one dimension (it applies to all values)"
        table_name)
      ();
    Some (List.hd units)

(** Scale a list of Ir.Const values from `unit` to the model's time unit.
    Non-Const entries (e.g. Param, BinOp) are passed through unchanged and
    a diagnostic is emitted — unit conversion of symbolic table values
    isn't implemented (would require re-materialising as BinOp { Mul, ... }
    which has knock-on dimcheck consequences). *)
let scale_table_values ctx ~table_name ~unit values =
  let scale = unit_to_model_time ctx 1.0 unit in
  if scale = 1.0 then values
  else List.map (fun v ->
    match v with
    | Ir.Const f -> Ir.Const (f *. scale)
    | other ->
      Diagnostics.error ctx.diags
        ~code:"E217"
        ~loc:Diagnostics.no_loc
        ~message:(Printf.sprintf
          "table '%s' has a '%s annotation but non-constant entries; \
           unit conversion of symbolic (parameter/expression) table \
           values isn't yet supported — declare values as plain \
           numbers or drop the unit annotation"
          table_name (unit_lit_to_string unit))
        ();
      other
  ) values

let table_dims ctx tname =
  match List.find_opt (fun td -> List.mem tname td.tnames) ctx.table_decls with
  | Some td -> List.map dim_name_of_entry td.tdims
  | None    -> []

(** Return the 0-based index of `value_name` within dimension
    `dim_name`'s ordered level list, as a float. Emits E263 + returns
    0 when the value isn't a level.

    Previously returned 0 silently on a miss (C2 in the 2026-04-19
    review), so `C_age[typo]` quietly resolved to `C_age[0]` — a
    stratified contact matrix with a typoed key silently used the
    wrong entry. Fix: emit a diagnostic naming the bad value and
    listing the valid levels. We still return 0 so downstream
    traversal can continue and surface any additional errors in a
    single pass; the diagnostic blocks compilation at exit.

    Levenshtein-distance "did you mean" hinting is possible but not
    implemented here — the levels list is small enough to eyeball. *)
let dim_value_index ctx dim_name value_name =
  let values = dim_values ctx dim_name in
  let rec find i = function
    | []                         -> None
    | v :: _ when v = value_name -> Some i
    | _ :: rest                  -> find (i + 1) rest
  in
  match find 0 values with
  | Some i -> float_of_int i
  | None ->
    Diagnostics.error ctx.diags
      ~code:"E263"
      ~loc:Diagnostics.no_loc
      ~message:(Printf.sprintf
        "'%s' is not a level of dimension '%s'" value_name dim_name)
      ~hint:(Printf.sprintf "valid levels: %s"
        (if values = [] then "(none)" else String.concat ", " values))
      ();
    0.0

(* ── Normalize expr ──────────────────────────────────────────────────────── *)

let rec normalize_expr (e : Ir.expr) : Ir.expr =
  match e with
  | Ir.BinOp { op = Ir.Add; left; right } -> (
    let l = normalize_expr left in
    let r = normalize_expr right in
    let rec collect_pops acc = function
      | Ir.Pop name  -> Some (name :: acc)
      | Ir.PopSum ps -> Some (List.rev_append ps acc)
      | Ir.BinOp { op = Ir.Add; left; right } -> (
          match collect_pops acc left with
          | Some acc' -> collect_pops acc' right
          | None -> None)
      | _ -> None
    in
    match collect_pops [] (Ir.BinOp { op = Ir.Add; left = l; right = r }) with
    | Some pops when List.length pops >= 2 -> Ir.PopSum (List.rev pops)
    | _ -> Ir.BinOp { op = Ir.Add; left = l; right = r }
  )
  | Ir.BinOp b ->
    let l = normalize_expr b.left in
    let r = normalize_expr b.right in
    Ir.BinOp { b with left = l; right = r }
  | Ir.UnOp u ->
    Ir.UnOp { u with arg = normalize_expr u.arg }
  | Ir.Cond c ->
    Ir.Cond { pred  = normalize_expr c.pred;
               then_ = normalize_expr c.then_;
               else_ = normalize_expr c.else_ }
  | other -> other

let ir_bin_op = function
  | Ast.Add -> Ir.Add | Ast.Sub -> Ir.Sub | Ast.Mul -> Ir.Mul
  | Ast.Div -> Ir.Div | Ast.Pow -> Ir.Pow
  | Ast.Eq  -> Ir.Eq  | Ast.Neq -> Ir.Neq
  | Ast.Lt  -> Ir.Lt  | Ast.Gt  -> Ir.Gt
  | Ast.Le  -> Ir.Le  | Ast.Ge  -> Ir.Ge

let ir_un_op = function
  | Ast.Neg   -> Ir.Neg  | Ast.Exp   -> Ir.Exp  | Ast.Log  -> Ir.Log
  | Ast.Sqrt  -> Ir.Sqrt | Ast.Abs   -> Ir.Abs  | Ast.Floor -> Ir.Floor
  | Ast.Ceil  -> Ir.Ceil
  | Ast.Sin   -> Ir.Sin  | Ast.Cos   -> Ir.Cos  | Ast.Tanh -> Ir.Tanh  (* gh#58 *)

(* ── Indexed parameter helpers ────────────────────────────────────────────── *)

(** True if [name] is the base name of an indexed parameter declaration. *)
let is_indexed_param ctx name =
  List.exists (fun pd ->
    match pd with
    | PIndexed p -> p.pname = name
    | _ -> false
  ) ctx.param_decls

(** True if [name] matches any fully-expanded indexed param (e.g. "R0_urban"). *)
let is_expanded_indexed_param_name ctx name =
  Hashtbl.mem ctx.expanded_param_tbl name

(** Resolve an index token in index position (inside [...]):
    1. Check substitution env  → stratum value via env binding
    2. Check if it is directly a member of any dimension → use as-is
    3. Otherwise → emit E100 and return the token unchanged *)
let resolve_index ctx (env : (string * string) list) idx =
  match List.assoc_opt idx env with
  | Some concrete -> concrete
  | None ->
    let all_vals = List.concat_map snd ctx.dim_registry in
    if List.mem idx all_vals then idx
    else begin
      Diagnostics.error ctx.diags
        ~code:"E100"
        ~loc:Diagnostics.no_loc
        ~message:(Printf.sprintf "unknown index value '%s'" idx)
        ~hint:"use a bound variable from [...] or a literal dimension member"
        ();
      idx  (* continue with placeholder *)
    end

(* ── Expression resolver ─────────────────────────────────────────────────── *)

let index_item_to_str env item =
  match item with
  | IPosn (EIdent (s, _))     -> (match List.assoc_opt s env with Some v -> v | None -> s)
  | IPosn _                   -> "?"
  | INamed (_, EIdent (s, _)) -> (match List.assoc_opt s env with Some v -> v | None -> s)
  | INamed (_, _)             -> "?"

(** Flatten a nested EList into a depth-first left-to-right list of leaf exprs. *)
let rec flatten_ast_list = function
  | EList es -> List.concat_map flatten_ast_list es
  | other    -> [other]

(** Compute the row-major flat index for a shaped let lookup.
    shape is the list of dimension names; items are the index arguments;
    env maps loop variable names to concrete level strings. *)
let shape_index ctx shape items env =
  let n = List.length shape in
  (* M20 in 2026-04-19 review: previously this called List.nth items i
     blindly — when `items` had fewer elements than `shape` (an under-
     applied shaped let), nth raised Failure("nth") which propagated
     unhandled through compile_detail_result's generic `exn -> Error`
     catch, and camdlc printed `Error: Failure("nth")` to the user. A
     compiler crash masquerading as a mysterious error. Fix: validate
     lengths up front with a proper diagnostic. *)
  if List.length items <> n then begin
    Diagnostics.error ctx.diags
      ~code:"E273"
      ~loc:Diagnostics.no_loc
      ~message:(Printf.sprintf
        "shaped let index has %d argument%s but the binding expects %d"
        (List.length items)
        (if List.length items = 1 then "" else "s")
        n)
      ~hint:(Printf.sprintf "shape dims: [%s]" (String.concat ", " shape))
      ();
    (* Fall through with 0s so downstream diagnostic-collection
       continues; the compile aborts at the end of this phase. *)
    0
  end else
  let pairs = List.mapi (fun i dim ->
    let item     = List.nth items i in
    let val_name = index_item_to_str env item in
    let idx      = int_of_float (dim_value_index ctx dim val_name) in
    let size     = List.length (dim_values ctx dim) in
    (idx, size)
  ) shape in
  (* Row-major: stride for dim i = product of sizes of dims i+1 ... n-1 *)
  let strides = Array.make n 1 in
  for i = n - 2 downto 0 do
    strides.(i) <- strides.(i + 1) * snd (List.nth pairs (i + 1))
  done;
  List.fold_left (fun acc (i, (idx, _)) -> acc + idx * strides.(i))
    0 (List.mapi (fun i p -> (i, p)) pairs)

(* Pure math functions that are safe to evaluate at compile time. *)
let is_const_func = function
  | "exp" | "log" | "sqrt" | "abs" | "floor" | "ceil" -> true
  | "sin" | "cos" | "tanh" -> true  (* gh#58 *)
  | _ -> false

let rec is_const_expr = function
  | EConst _ | EUnit _ -> true
  | EUnOp (_, e) -> is_const_expr e
  | EBinOp (_, l, r) -> is_const_expr l && is_const_expr r
  | EFuncCall (fname, args) when is_const_func fname ->
    (* Pure math functions are const-foldable iff all args are const.
       The parser emits EFuncCall for log/exp/sqrt/etc.; EUnOp(Log,_) is
       dead unless another AST-level pass rewrites them. *)
    List.for_all (fun (_, e) -> is_const_expr e) args
  | _ -> false

let rec resolve_expr ctx (env : (string * string) list) (e : expr) : Ir.expr =
  match e with
  | EConst f     -> Ir.Const f
  | EUnit (f, u) ->
    (* Preserve the unit's dimension on the IR so Dimcheck can see it.
       Duration units (`'days` etc.) wrap as UncheckedDim with dim = T;
       rate units as UncheckedDim with dim = T⁻¹. The scalar is pre-
       scaled into the model's time unit. For Count/Ratio we keep the
       existing bare-Const behaviour since there is no time scale.
       Fixes GH #9: previously `1 'days` became a bare dimensionless
       Const, so `(gamma + mu) * 1 'days` type-checked as T⁻¹ and
       `exp()` of it produced a spurious E301. *)
    let scaled = unit_to_model_time ctx f u in
    (match u with
     | Count | Ratio -> Ir.Const scaled
     | _ ->
       let (p, t) = unit_lit_to_dim u in
       Ir.UncheckedDim {
         inner  = Ir.Const scaled;
         dim_p  = p;
         dim_t  = t;
         reason = Printf.sprintf "unit literal '%s" (unit_lit_to_string u);
       })
  | EIdent (name, l) -> (
    let loc = diag_loc_of_ast_ctx ctx l in
    match List.assoc_opt name env with
    | Some concrete -> resolve_ident_name ctx concrete ~loc
    | None          -> resolve_ident_name ctx name ~loc
  )
  | EIndex (name, items) -> (
    let base_name =
      match List.assoc_opt name env with Some n -> n | None -> name
    in
    (* 1. Table? → TableLookup with a single flattened linear index.
       For a table of dims [d1; d2; ...] with sizes [n1; n2; ...], the
       linear index is: i1*n2*n3*... + i2*n3*... + ... + iN.
       The IR and Rust runtime always expect exactly one index. *)
    let tdims = table_dims ctx base_name in
    if tdims <> [] then
      let per_dim = List.mapi (fun i item ->
        let dim      = List.nth tdims i in
        let val_name = index_item_to_str env item in
        (int_of_float (dim_value_index ctx dim val_name),
         List.length (dim_values ctx dim))
      ) items in
      (* stride for dimension i = product of sizes of all later dimensions *)
      let n = List.length per_dim in
      let linear = List.fold_left (fun (acc, pos) (idx, _) ->
        let stride = List.fold_left (fun s j ->
          s * snd (List.nth per_dim j)
        ) 1 (List.init (n - pos - 1) (fun k -> pos + 1 + k)) in
        (acc + idx * stride, pos + 1)
      ) (0, 0) per_dim |> fst in
      Ir.TableLookup (base_name, [Ir.Const (float_of_int linear)])
    else
    (* 2. Indexed let binding? → inline body with index vars substituted *)
    match Hashtbl.find_opt ctx.let_tbl base_name with
    | Some lb when lb.lindices <> [] ->
      let inner_env = List.mapi (fun i ib ->
        let var_name = match ib with
          | IBind (v, _)      -> v
          | IConsec (v, _, _) -> v
          | IComp v           -> v
        in
        let val_name = match List.nth_opt items i with
          | Some item -> index_item_to_str env item
          | None      -> "?"
        in
        (var_name, val_name)
      ) lb.lindices in
      normalize_expr (resolve_expr ctx (inner_env @ env) lb.lbody)
    (* 2b. Shaped let? → flatten body, compute row-major index, resolve cell *)
    | Some lb when lb.lshape <> None ->
      let shape = Option.get lb.lshape in
      let flat  = flatten_ast_list lb.lbody in
      let idx   = shape_index ctx shape items env in
      if idx >= 0 && idx < List.length flat then
        normalize_expr (resolve_expr ctx env (List.nth flat idx))
      else begin
        Diagnostics.error ctx.diags
          ~code:"E218" ~loc:Diagnostics.no_loc
          ~message:(Printf.sprintf
            "shaped let '%s': index %d out of bounds (size %d)"
            base_name idx (List.length flat)) ();
        Ir.Const 0.0
      end
    | _ ->
    (* 2c. Indexed time function: beta[p] → Ir.TimeFunc "beta_urban" *)
    if (match Hashtbl.find_opt ctx.func_tbl base_name with
        | Some fd -> fd.findices <> [] | None -> false) then
      let idx_vals = List.map (index_item_to_str env) items in
      Ir.TimeFunc (String.concat "_" (base_name :: idx_vals))
    else
    (* 3. Indexed parameter? → resolve index and return Ir.Param of mangled name *)
    if is_indexed_param ctx base_name then
      (match items with
       | [IPosn (EIdent (idx, _))] | [INamed (_, EIdent (idx, _))] ->
         let concrete = resolve_index ctx env idx in
         Ir.Param (base_name ^ "_" ^ concrete)
       | _ ->
         (* multi-item or non-ident index: fall through to name mangling *)
         let idx_vals = List.map (index_item_to_str env) items in
         let concrete = String.concat "_" (base_name :: idx_vals) in
         resolve_ident_name ctx concrete ~loc:Diagnostics.no_loc)
    else
    (* 4. Compartment with indices → concatenate to concrete name *)
    let idx_vals = List.map (index_item_to_str env) items in
    let concrete = String.concat "_" (base_name :: idx_vals) in
    resolve_ident_name ctx concrete ~loc:Diagnostics.no_loc
  )
  | EBinOp (op, l, r) ->
    let ir_l = resolve_expr ctx env l in
    let ir_r = resolve_expr ctx env r in
    normalize_expr (Ir.BinOp { op = ir_bin_op op; left = ir_l; right = ir_r })
  | EUnOp (op, e) ->
    Ir.UnOp { op = ir_un_op op; arg = resolve_expr ctx env e }
  | ECond (p, a, b) ->
    Ir.Cond { pred  = resolve_expr ctx env p;
               then_ = resolve_expr ctx env a;
               else_ = resolve_expr ctx env b }
  | ESum (v, d, body) ->
    let vals = dim_values ctx d in
    if vals = [] then Ir.Const 0.0
    else
      let terms = List.map (fun vv ->
        resolve_expr ctx ((v, vv) :: env) body
      ) vals in
      (* Use plain Add — do NOT normalize here; normalize_expr only collapses
         all-Pop Add-trees, but sum terms are typically Mul-trees. *)
      List.fold_left (fun acc t ->
        Ir.BinOp { op = Ir.Add; left = acc; right = t }
      ) (List.hd terms) (List.tl terms)
  | EFuncCall ("date", args) ->
    let date_str = match args with
      | [("", EIdent (s, _))] -> s
      | _ ->
        Diagnostics.error ctx.diags ~code:"E220" ~loc:Diagnostics.no_loc
          ~message:"date() expects a single quoted string argument, e.g. date(\"2020-01-01\")"
          ();
        ""
    in
    (match ctx.origin with
     | Some origin_str ->
       (try Ir.Const (parse_date_to_float origin_str date_str ctx.time_unit)
        with Failure msg ->
          Diagnostics.error ctx.diags ~code:"E220" ~loc:Diagnostics.no_loc
            ~message:msg ();
          Ir.Const 0.0)
     | None ->
       Diagnostics.error ctx.diags ~code:"E220" ~loc:Diagnostics.no_loc
         ~message:"date() requires a top-level origin declaration, e.g. origin = date(\"2020-01-01\")"
         ();
       Ir.Const 0.0)
  | EFuncCall ("unchecked_dim", args) ->
    (* Per-expression dimensional escape. Grammar:
         unchecked_dim(<expr>, dim = <name>, reason = "why")
       where <name> is one of: dimensionless, population, time, rate,
       population_rate, per_population. Compiles to `Ir.UncheckedDim`
       with the inner expression resolved recursively and the asserted
       (P, T) dimension packed from the name. The dim-checker uses
       the declared dim authoritatively and does NOT unify the inner
       expression's dim — that's the whole point of the escape. *)
    let dim_of_name n = match n with
      | "dimensionless"    -> Some (0, 0)
      | "population"       -> Some (1, 0)
      | "time"             -> Some (0, 1)
      | "rate"             -> Some (0, -1)
      | "population_rate"  -> Some (1, -1)
      | "per_population"   -> Some (-1, 0)
      | _ -> None
    in
    let get_kw k = List.assoc_opt k args in
    let positional = List.filter_map (fun (k, e) -> if k = "" then Some e else None) args in
    let inner_expr = match positional with
      | [e] -> e
      | _ ->
        Diagnostics.error ctx.diags ~code:"E240"
          ~loc:Diagnostics.no_loc
          ~message:"unchecked_dim takes exactly one positional argument"
          ~hint:"usage: unchecked_dim(expr, dim = <name>, reason = \"...\")" ();
        EConst 0.0
    in
    let dim_name = match get_kw "dim" with
      | Some (EIdent (n, _)) -> n
      | Some _ | None ->
        Diagnostics.error ctx.diags ~code:"E240"
          ~loc:Diagnostics.no_loc
          ~message:"unchecked_dim requires `dim = <name>` keyword argument"
          ~detail:"Domain names: dimensionless, population, time, rate, population_rate, per_population."
          ~hint:"example: unchecked_dim((I + iota)^alpha, dim = population, reason = \"He et al. 2010 α-mixing\")"
          ();
        "dimensionless"
    in
    let (dim_p, dim_t) = match dim_of_name dim_name with
      | Some pair -> pair
      | None ->
        Diagnostics.error ctx.diags ~code:"E240"
          ~loc:Diagnostics.no_loc
          ~message:(Printf.sprintf "unchecked_dim: unknown dim name '%s'" dim_name)
          ~detail:"Valid: dimensionless, population, time, rate, population_rate, per_population."
          ();
        (0, 0)
    in
    let reason = match get_kw "reason" with
      | Some (EIdent (s, _)) -> s
      | Some _ | None ->
        Diagnostics.error ctx.diags ~code:"E240"
          ~loc:Diagnostics.no_loc
          ~message:"unchecked_dim requires `reason = \"...\"` string kwarg"
          ~detail:"A string documenting why the dimensional assertion is legitimate. \
                   Escape hatches must be justified at the call site; see \
                   docs/dev/proposals/notes/unchecked-dim-escape.md."
          ~hint:"reason = \"He et al. 2010 α-mixing exponent\""
          ();
        "<no reason given>"
    in
    Ir.UncheckedDim {
      Ir.inner  = resolve_expr ctx env inner_expr;
      Ir.dim_p; Ir.dim_t; Ir.reason;
    }
  | EFuncCall (fname, args) ->
    (* Built-in math functions → Ir.UnOp *)
    let builtin_un_op = match fname with
      | "exp" -> Some Ir.Exp | "log" -> Some Ir.Log | "sqrt" -> Some Ir.Sqrt
      | "abs" -> Some Ir.Abs | "floor" -> Some Ir.Floor | "ceil" -> Some Ir.Ceil
      | "sin" -> Some Ir.Sin | "cos" -> Some Ir.Cos | "tanh" -> Some Ir.Tanh  (* gh#58 *)
      | _ -> None in
    if Option.is_some builtin_un_op then begin
      let op = Option.get builtin_un_op in
      match args with
      | [("", arg)] -> Ir.UnOp { op; arg = resolve_expr ctx env arg }
      | _ ->
        Diagnostics.error ctx.diags ~code:"E101" ~loc:Diagnostics.no_loc
          ~message:(Printf.sprintf "built-in function '%s' takes exactly one argument" fname)
          ~hint:(Printf.sprintf "usage: %s(expr)" fname) ();
        Ir.Const 0.0
    end
    else if fname = "mod" then begin
      match args with
      | [("", a); ("", b)] ->
        Ir.BinOp { op = Ir.Mod; left = resolve_expr ctx env a; right = resolve_expr ctx env b }
      | _ ->
        Diagnostics.error ctx.diags ~code:"E101" ~loc:Diagnostics.no_loc
          ~message:"built-in function 'mod' takes exactly two arguments"
          ~hint:"usage: mod(a, b)" ();
        Ir.Const 0.0
    end
    else if Hashtbl.mem ctx.func_tbl fname
    then begin
      let ok = match args with
        | [] -> true                                       (* bare: seasonal *)
        | [("", EIdent ("t", _))] -> true                  (* explicit: seasonal(t) *)
        | _ -> false
      in
      if not ok then
        Diagnostics.error ctx.diags ~code:"E101" ~loc:Diagnostics.no_loc
          ~message:(Printf.sprintf "forcing function '%s' takes no arguments, or (t) for the current simulation time" fname)
          ~hint:(Printf.sprintf "write '%s' or '%s(t)'" fname fname) ();
      Ir.TimeFunc fname
    end
    else begin
      Diagnostics.error ctx.diags ~code:"E100" ~loc:Diagnostics.no_loc
        ~message:(Printf.sprintf "undeclared function '%s'" fname)
        ~hint:"check spelling, or add a declaration in forcing { }" ();
      Ir.Const 0.0
    end
  | EList _     ->
    (* m17 in 2026-04-19 review: list literals are only valid as table
       values, scheduled `at = [...]` times, or periodic `on = [...]`
       specs — they have no meaning in a scalar rate expression.
       Previously silently returned Const 0.0, which meant any use of
       a list in the wrong context gave `rate = 0` and no diagnostic. *)
    Diagnostics.error ctx.diags ~code:"E270" ~loc:Diagnostics.no_loc
      ~message:"list literal not allowed in a scalar expression"
      ~hint:"lists are valid as: table values, `at = [...]` times, \
             or `on = [...]` periodic specs"
      ();
    Ir.Const 0.0
  | ERange _    ->
    Diagnostics.error ctx.diags ~code:"E271" ~loc:Diagnostics.no_loc
      ~message:"range expression not allowed in a scalar expression"
      ~hint:"ranges are only valid inside `periodic on = [...]`"
      ();
    Ir.Const 0.0

and resolve_ident_name ctx name ~loc =
  (* 1. Let binding? Inline it — unless it's a typed const (emitted as Param). *)
  match Hashtbl.find_opt ctx.let_tbl name with
  | Some lb ->
    if lb.lkind <> None && is_const_expr lb.lbody then
      (* Typed const let → treat as parameter (dimcheck will see param_kind) *)
      Ir.Param name
    else
      normalize_expr (resolve_expr ctx [] lb.lbody)
  | None ->
  (* 2. Known expanded compartment? *)
  if Hashtbl.mem ctx.expanded_comp_tbl name then Ir.Pop name
  else if Hashtbl.mem ctx.comp_tbl name then begin
    let expansions = expand_compartment_name ctx name in
    if List.length expansions = 1 then Ir.Pop (List.hd expansions)
    else Ir.PopSum expansions
  end
  else if Hashtbl.mem ctx.scalar_param_tbl name then
    Ir.Param name
  else if is_expanded_indexed_param_name ctx name then
    Ir.Param name
  else if Hashtbl.mem ctx.func_tbl name then
    Ir.TimeFunc name
  else if name = "t" then
    Ir.Time
  else if name = "dt" then
    (* gh#54: runtime integrator step. Has dimension T (same as `t`).
       Backend evaluates against `EvalCtx.dt` populated from
       SMCConfig / ChainBinomialConfig at substep level. *)
    Ir.Dt
  else if name = "projected" then
    (* Special keyword in likelihood expressions: refers to the observation projection output. *)
    Ir.Projected
  else if name = "pi" then
    (* gh#58: mathematical constant. Resolves to Ir.Const at expand time —
       no new IR variant. *)
    Ir.Const Float.pi
  else if name = "e" then
    Ir.Const (Float.exp 1.0)
  else begin
    Diagnostics.error ctx.diags
      ~code:"E100"
      ~loc
      ~message:(Printf.sprintf "undeclared name '%s'" name)
      ~hint:"check spelling, or add a declaration in compartments/parameters/let/tables"
      ();
    Ir.Const 0.0  (* placeholder — compilation continues to collect more errors *)
  end

(* ── Stoichiometry ────────────────────────────────────────────────────────── *)

(** Resolve a stoichiometry reference to a fully-qualified compartment
    name. When the base has multiple stratified expansions but the
    reference has no indices, previously returned the bare base name
    (C3 in the 2026-04-19 review) — producing `("S", 1)` in IR
    stoichiometry for a model where S was stratified into [S_child,
    S_adult]. The bare name `S` isn't in the expanded compartments
    list, so the emitted IR was structurally invalid.

    Now: error out naming the transition's compartment and listing
    the valid expansions. Still returns `base` as a continuation so
    the caller can emit additional diagnostics before the compile
    aborts. *)
let resolve_stoich_ref ctx env (cname, items) =
  let base = match List.assoc_opt cname env with Some n -> n | None -> cname in
  let idx_vals = List.map (index_item_to_str env) items in
  if idx_vals = [] then begin
    let expansions = expand_compartment_name ctx base in
    match expansions with
    | [single] -> single
    | [] -> base  (* unknown compartment — caught downstream by Validate *)
    | many ->
      Diagnostics.error ctx.diags
        ~code:"E272"
        ~loc:Diagnostics.no_loc
        ~message:(Printf.sprintf
          "compartment '%s' is stratified but used without indices in \
           stoichiometry" base)
        ~hint:(Printf.sprintf
          "pick an expansion or index the transition: %s"
          (String.concat ", " many))
        ();
      base
  end else
    String.concat "_" (base :: idx_vals)

(* ── Origin kind inference ───────────────────────────────────────────────── *)

let contains_pop_other_than expr src_name =
  let found = ref false in
  let rec walk = function
    | Ir.Pop n          -> if n <> src_name then found := true
    | Ir.PopSum ns      -> if List.exists (fun n -> n <> src_name) ns then found := true
    | Ir.BinOp b        -> walk b.left; walk b.right
    | Ir.UnOp u         -> walk u.arg
    | Ir.Cond c         -> walk c.pred; walk c.then_; walk c.else_
    | Ir.TableLookup (_, idxs) -> List.iter walk idxs
    | _                 -> ()
  in
  walk expr; !found

let infer_origin_kind src_opt dst_opt rate =
  match src_opt, dst_opt with
  | None,      _       -> "inflow"
  | _,         None    -> "outflow"
  | Some src,  Some _  ->
    if contains_pop_other_than rate src then "transmission"
    else "intrinsic"

(* ── Cartesian product of index bindings ─────────────────────────────────── *)

let cartesian_product ibs ctx =
  let axes = List.filter_map (fun ib ->
    match ib with
    | IBind (v, d) ->
      let vals = dim_values ctx d in
      if vals = [] then None
      else Some (List.map (fun vv -> [(v, vv)]) vals)
    | IConsec (v, vn, d) ->
      let vals = dim_values ctx d in
      let n = List.length vals in
      if n < 2 then None
      else begin
        (* Only generate pairs for valid consecutive positions i < n-1 *)
        let pairs = List.filteri (fun i _ -> i < n - 1) vals
          |> List.mapi (fun i vv ->
               let vv_next = List.nth vals (i + 1) in
               [(v, vv); (vn, vv_next)])
        in
        if pairs = [] then None else Some pairs
      end
    | IComp v ->
      (* Iterate over all base compartment names (Integer kind only) *)
      let names = List.filter_map (fun cd ->
        match cd.ckind with
        | Integer -> Some cd.cname
        | Real    -> None
      ) ctx.comp_decls in
      if names = [] then None
      else Some (List.map (fun n -> [(v, n)]) names)
  ) ibs in
  if axes = [] then [[]]
  else begin
    let rec cart = function
      | [] -> [[]]
      | ax :: rest ->
        let tails = cart rest in
        List.concat_map (fun binds ->
          List.map (fun tail -> binds @ tail) tails
        ) ax
    in
    cart axes
  end

(* ── Transition name helpers ─────────────────────────────────────────────── *)

(** Extract the name-suffix parts from index bindings in order.
    For IBind/IComp use the bound variable's value; for IConsec use only
    the first variable's value (not a_next). *)
let name_parts_from_bindings ibs env =
  List.filter_map (fun ib ->
    match ib with
    | IBind (v, _)      -> List.assoc_opt v env
    | IConsec (v, _, _) -> List.assoc_opt v env
    | IComp v           -> List.assoc_opt v env
  ) ibs

(* ── Guard evaluation ─────────────────────────────────────────────────────── *)

let rec eval_guard env = function
  | GEq (a, b) ->
    let va = Option.value ~default:a (List.assoc_opt a env) in
    let vb = Option.value ~default:b (List.assoc_opt b env) in
    va = vb
  | GNeq (a, b) ->
    let va = Option.value ~default:a (List.assoc_opt a env) in
    let vb = Option.value ~default:b (List.assoc_opt b env) in
    va <> vb
  | GAnd (g1, g2) -> eval_guard env g1 && eval_guard env g2
  | GOr  (g1, g2) -> eval_guard env g1 || eval_guard env g2

(** Validate that every identifier in a guard is either a loop variable,
    a dimension level value, or an unknown name — but NOT a parameter or
    compartment name (which cannot be meaningfully compared at compile time).
    Emits E217 for each bad identifier found. *)
let check_guard_compile_time ?(loc = Diagnostics.no_loc) ctx decl_name loop_vars guard =
  let all_dim_levels = List.concat_map snd ctx.dim_registry in
  let param_names = List.filter_map (function
    | PScalar  p -> Some p.pname
    | PIndexed p -> Some p.pname
  ) ctx.param_decls in
  let comp_names = List.map (fun c -> c.cname) ctx.comp_decls in
  let check_ident ident =
    if List.mem ident loop_vars || List.mem ident all_dim_levels then ()
    else if List.mem ident param_names then
      Diagnostics.error ctx.diags
        ~code:"E217" ~loc
        ~message:(Printf.sprintf
          "%s: where guard references '%s', which is a parameter; \
           use it in the rate expression instead"
          decl_name ident) ()
    else if List.mem ident comp_names then
      Diagnostics.error ctx.diags
        ~code:"E217" ~loc
        ~message:(Printf.sprintf
          "%s: where guard references '%s', which is a compartment; \
           use it in the rate expression instead"
          decl_name ident) ()
  in
  let rec walk = function
    | GEq (a, b) | GNeq (a, b) -> check_ident a; check_ident b
    | GAnd (g1, g2) | GOr (g1, g2) -> walk g1; walk g2
  in
  walk guard

let loop_vars_of_indices indices =
  List.concat_map (function
    | IBind (v, _)       -> [v]
    | IConsec (v, vn, _) -> [v; vn]
    | IComp v            -> [v]
  ) indices

(** Check all transition and intervention guards for E217 (non-evaluable idents). *)
let check_guards ctx =
  List.iter (fun tr ->
    match tr.trguard with
    | None -> ()
    | Some g ->
      check_guard_compile_time ctx ~loc:(diag_loc_of_ast_ctx ctx tr.trloc)
        tr.trname (loop_vars_of_indices tr.trindices) g
  ) ctx.transitions;
  List.iter (fun iv ->
    match iv.ivguard with
    | None -> ()
    | Some g ->
      check_guard_compile_time ctx iv.ivname
        (loop_vars_of_indices iv.ivindices) g
  ) ctx.interv_decls

(* ── Transition expansion ────────────────────────────────────────────────── *)

let guard_to_string g =
  let rec pp = function
    | GEq  (a, b) -> Printf.sprintf "%s == %s" a b
    | GNeq (a, b) -> Printf.sprintf "%s != %s" a b
    | GAnd (g1, g2) -> Printf.sprintf "%s and %s" (pp g1) (pp g2)
    | GOr  (g1, g2) -> Printf.sprintf "%s or %s"  (pp g1) (pp g2)
  in
  pp g

let expand_transitions_counted ctx =
  let filtered = ref 0 in
  let expanded = List.concat_map (fun tr ->
    let combos = cartesian_product tr.trindices ctx in
    let tr_filtered = ref 0 in
    let results = List.map (fun env ->
      let pass_guard = match tr.trguard with
        | None   -> true
        | Some g -> eval_guard env g
      in
      if not pass_guard then (incr filtered; incr tr_filtered; [])
      else begin
        let src_names = List.map (resolve_stoich_ref ctx env) tr.trsrc in
        (* Extract rate wrappers: overdispersed(rate, σ²) or
           deterministic(rate). Mismatched arg shapes are a hard
           error (reported as C1 in the 2026-04-19 review) — before
           this, any shape other than the exact positional form fell
           through to `_ -> DrawPoisson`, so users who wrote
           `overdispersed(rate=foo, sigma=bar)` or `overdispersed(foo)`
           silently got a pure Poisson draw with no diagnostic.
           Inference under the wrong noise model produced biased
           posteriors; this is the "silent wrong answer" class. *)
        let validate_draw_shape name args n_expected shape_hint =
          Diagnostics.error ctx.diags
            ~code:"E260"
            ~loc:Diagnostics.no_loc
            ~message:(Printf.sprintf
              "%s() takes %d positional argument%s: %s"
              name n_expected
              (if n_expected = 1 then "" else "s") shape_hint)
            ~hint:(Printf.sprintf
              "saw %d argument%s%s"
              (List.length args)
              (if List.length args = 1 then "" else "s")
              (if List.exists (fun (k, _) -> k <> "") args
               then " (keyword args not supported here — use positional)"
               else ""))
            ()
        in
        let raw_rate, draw_method = match tr.trrate with
          | EFuncCall ("overdispersed", [("", inner); ("", var)]) ->
            let resolved_var = normalize_expr (resolve_expr ctx env var) in
            (inner, Ir.DrawOverdispersed resolved_var)
          | EFuncCall ("overdispersed", args) ->
            validate_draw_shape "overdispersed" args 2
              "overdispersed(rate, sigma_squared)";
            (tr.trrate, Ir.DrawPoisson)
          | EFuncCall ("deterministic", [("", inner)]) ->
            (inner, Ir.DrawDeterministic)
          | EFuncCall ("deterministic", args) ->
            validate_draw_shape "deterministic" args 1
              "deterministic(rate)";
            (tr.trrate, Ir.DrawPoisson)
          | _ -> (tr.trrate, Ir.DrawPoisson)
        in
        (* Build one IR transition given resolved destinations and a
           (possibly weight-scaled) raw rate. `name_suffix` gets
           appended to the transition name — used for branches to
           disambiguate `infect` → `infect_symp` / `infect_asym`. *)
        let emit_one dst_refs raw_rate_for_branch name_suffix =
          let dst_names = List.map (resolve_stoich_ref ctx env) dst_refs in
          let rate = normalize_expr (resolve_expr ctx env raw_rate_for_branch) in
          let raw_entries =
            List.map (fun n -> (n, -1)) src_names
            @ List.map (fun n -> (n,  1)) dst_names
          in
          let collapse entries =
            let order = ref [] in
            let tbl = Hashtbl.create 8 in
            List.iter (fun (n, d) ->
              if not (Hashtbl.mem tbl n) then order := n :: !order;
              let prev = try Hashtbl.find tbl n with Not_found -> 0 in
              Hashtbl.replace tbl n (prev + d)
            ) entries;
            List.filter_map (fun n ->
              let d = Hashtbl.find tbl n in
              if d = 0 then None else Some (n, d)
            ) (List.rev !order)
          in
          let stoich = collapse raw_entries in
          let sole_with_sign sign =
            let matches = List.filter (fun (_, d) ->
              if sign < 0 then d < 0 else d > 0) stoich in
            match matches with
            | [(n, _)] -> Some n
            | _        -> None
          in
          let src_meta = sole_with_sign (-1) in
          let dst_meta = sole_with_sign   1  in
          if stoich = [] && (src_names <> [] || dst_names <> []) then begin
            Diagnostics.error ctx.diags
              ~code:"E310"
              ~loc:Diagnostics.no_loc
              ~message:(Printf.sprintf
                "transition '%s' has no net effect: sources and destinations cancel"
                tr.trname)
              ~hint:"remove catalyst compartments that appear on both sides, \
                     or declare the transition with a non-trivial net stoichiometry"
              ()
          end;
          let origin_kind = infer_origin_kind src_meta dst_meta rate in
          let parts = name_parts_from_bindings tr.trindices env in
          let base_name =
            if parts = [] then tr.trname
            else tr.trname ^ "_" ^ String.concat "_" parts
          in
          let tr_name = match name_suffix with
            | None -> base_name
            | Some s -> base_name ^ "_" ^ s
          in
          (* Lineage analysis (#[lineage], 2026-05-19 proposal). Runs on
             the resolved/normalized IR rate so Pop names are fully
             qualified after stratification, and the source set is the
             expanded source compartments. A nonlinear use of a parent
             count is rejected with E601 pointing at the transition. *)
          let lineage =
            if not tr.trlineage then None
            else begin
              let cls = Lineage.classify_parents ~sources:src_names rate in
              match cls.Lineage.nonlinear with
              | Some nl ->
                Diagnostics.error ctx.diags
                  ~code:"E601"
                  ~loc:(diag_loc_of_ast_ctx ctx tr.trloc)
                  ~message:(Printf.sprintf
                    "lineage tracking on transition '%s' requires linear \
                     dependence on parent compartments. Found nonlinear use \
                     of '%s' in the rate expression (inside %s)."
                    tr_name nl.Lineage.comp
                    (Pp_expr.to_string nl.Lineage.context))
                  ~hint:"options: (1) rewrite the rate so the parent appears \
                         as a top-level linear factor, absorbing the \
                         nonlinearity into other parameters; (2) remove the \
                         #[lineage] annotation — v1 lineage tracking does not \
                         support nonlinear parent dependence; (3) wait for \
                         Phase 4 lineage support (nonlinear rates with \
                         explicit attribution semantics)."
                  ();
                (* Emit a structurally-valid lineage record so compilation
                   continues to collect further diagnostics; the error above
                   blocks a successful compile. *)
                Some { Ir.is_lineage_event = true; Ir.parent_pool_weights = [] }
              | None ->
                let weights = Lineage.parent_pool_weights ~sources:src_names rate in
                Some { Ir.is_lineage_event = true; Ir.parent_pool_weights = weights }
            end
          in
          {
            Ir.name            = tr_name;
            Ir.stoichiometry   = stoich;
            Ir.rate            = rate;
            Ir.metadata        = Some {
              Ir.origin_kind        = Some origin_kind;
              Ir.source_compartment = src_meta;
              Ir.dest_compartment   = dst_meta;
            };
            Ir.draw_method     = draw_method;
            Ir.rate_grad       = [];  (* populated later by autodiff pass *)
            Ir.lineage         = lineage;
          }
        in
        (* Dispatch on destination form.
           - DstSum: one emitted transition per combo (classic path).
           - DstBranch: one transition per branch, with rate = weight_i
             * raw_rate. The suffix is derived from the branch's
             destination compartment (pre-stratification) so the final
             transition names are stable across index expansion. *)
        match tr.trdst with
        | DstSum dsts -> [emit_one dsts raw_rate None]
        | DstBranch branches ->
          List.map (fun ((dst_ref, weight) : (Ast.stoich_ref * Ast.expr)) ->
            let (dst_base, _idx) = dst_ref in
            let scaled_rate = Ast.EBinOp (Ast.Mul, weight, raw_rate) in
            emit_one [dst_ref] scaled_rate (Some dst_base)
          ) branches
      end
    ) combos in
    let results = List.concat results in
    (* Warn if a where guard filtered ALL combinations to zero transitions *)
    (match tr.trguard with
     | Some g when results = [] && combos <> [] ->
       Diagnostics.warning ctx.diags
         ~code:"W200" ~loc:Diagnostics.no_loc
         ~message:(Printf.sprintf
           "'where' guard in transition '%s' produced 0 transitions"
           tr.trname)
         ~detail:(Printf.sprintf
           "The guard `where %s` filtered all %d combinations."
           (guard_to_string g) (List.length combos))
         ~hint:"Check that the guard variable names match the loop variables."
         ()
     | _ -> ());
    results
  ) ctx.transitions in
  (expanded, !filtered)

let expand_transitions ctx =
  fst (expand_transitions_counted ctx)

(* ── Parameter expansion ─────────────────────────────────────────────────── *)

let resolve_float_expr_simple ctx e =
  let ir = normalize_expr (resolve_expr ctx [] e) in
  match ir with
  | Ir.Const f -> f
  | _ -> 0.0

let resolve_bounds ctx pbounds =
  match pbounds with
  | None -> None
  | Some (lo_e, hi_e) ->
    let lo = resolve_float_expr_simple ctx lo_e in
    let hi = resolve_float_expr_simple ctx hi_e in
    Some (lo, hi)

let param_kind_to_string = function
  | PRate        -> "rate"
  | PProbability -> "probability"
  | PPositive    -> "positive"
  | PCount       -> "count"
  | PReal        -> "real"

let rec eval_const_expr ctx = function
  | EConst f -> f
  | EUnit (f, u) -> unit_to_model_time ctx f u
  | EUnOp (Neg, e) -> -. (eval_const_expr ctx e)
  | EUnOp (Exp, e) -> exp (eval_const_expr ctx e)
  | EUnOp (Log, e) -> log (eval_const_expr ctx e)
  | EUnOp (Sqrt, e) -> sqrt (eval_const_expr ctx e)
  | EUnOp (Abs, e) -> abs_float (eval_const_expr ctx e)
  | EUnOp (Floor, e) -> floor (eval_const_expr ctx e)
  | EUnOp (Ceil, e) -> ceil (eval_const_expr ctx e)
  | EUnOp (Sin, e)  -> sin (eval_const_expr ctx e)
  | EUnOp (Cos, e)  -> cos (eval_const_expr ctx e)
  | EUnOp (Tanh, e) -> tanh (eval_const_expr ctx e)
  | EBinOp (Add, l, r) -> eval_const_expr ctx l +. eval_const_expr ctx r
  | EBinOp (Sub, l, r) -> eval_const_expr ctx l -. eval_const_expr ctx r
  | EBinOp (Mul, l, r) -> eval_const_expr ctx l *. eval_const_expr ctx r
  | EBinOp (Div, l, r) -> eval_const_expr ctx l /. eval_const_expr ctx r
  | EBinOp (Pow, l, r) -> eval_const_expr ctx l ** eval_const_expr ctx r
  | EFuncCall (fname, [(_, e)]) when is_const_func fname ->
    let v = eval_const_expr ctx e in
    (match fname with
     | "exp"   -> exp v
     | "log"   -> log v
     | "sqrt"  -> sqrt v
     | "abs"   -> abs_float v
     | "floor" -> floor v
     | "ceil"  -> ceil v
     | "sin"   -> sin v
     | "cos"   -> cos v
     | "tanh"  -> tanh v
     | _       -> 0.0 (* unreachable — is_const_func filters these *))
  | _ -> 0.0  (* unreachable — guarded by is_const_expr *)

(* Full resolve_float_expr: tries AST const-eval first, then IR reduction.
   Errors if neither produces a constant. *)
let resolve_float_expr ctx e =
  if is_const_expr e then eval_const_expr ctx e
  else
    let ir = normalize_expr (resolve_expr ctx [] e) in
    match ir with
    | Ir.Const f -> f
    | _ ->
      Diagnostics.error ctx.diags
        ~code:"E401" ~loc:Diagnostics.no_loc
        ~message:"expected a constant expression"
        ~detail:"This position requires a compile-time constant (number or \
                 arithmetic of constants). Parameters and compartments are \
                 not allowed here."
        ~hint:"Use a numeric literal or arithmetic of literals."
        ();
      0.0

(* ── Prior distribution resolution ─────────────────────────────────────── *)

(** Expected keyword arguments for each supported prior distribution.
    The first element of each pair is the arg name, the second is a
    value-validator returning [Some error_msg] on failure. *)
let prior_arg_signature = function
  | "uniform"     -> Some ["lower"; "upper"]
  | "normal"      -> Some ["mu"; "sigma"]
  | "log_normal"  -> Some ["mu"; "sigma"]
  | "half_normal" -> Some ["sigma"]
  | "beta"        -> Some ["alpha"; "beta"]
  | "gamma"       -> Some ["shape"; "rate"]
  | "exponential" -> Some ["rate"]
  | _             -> None

(** Per-distribution value validation. Returns [Some msg] if the
    argument bundle violates a distributional constraint. *)
let validate_prior_values dist_name vals =
  let find k = List.assoc_opt k vals in
  let pos_check key =
    match find key with
    | Some v when v <= 0.0 ->
      Some (Printf.sprintf "argument '%s' must be positive (got %g)" key v)
    | _ -> None
  in
  match dist_name with
  | "uniform" ->
    (match find "lower", find "upper" with
     | Some lo, Some hi when lo >= hi ->
       Some (Printf.sprintf "uniform requires lower < upper (got lower=%g, upper=%g)" lo hi)
     | _ -> None)
  | "normal" | "log_normal" -> pos_check "sigma"
  | "half_normal" -> pos_check "sigma"
  | "beta" ->
    (match pos_check "alpha" with Some _ as e -> e | None -> pos_check "beta")
  | "gamma" ->
    (match pos_check "shape" with Some _ as e -> e | None -> pos_check "rate")
  | "exponential" -> pos_check "rate"
  | _ -> None

type prior_classification =
  [ `Plain        of Ir.prior_dist
  | `Hierarchical of Ir.hierarchical_prior ]

let resolve_prior_spec ?(loc = Diagnostics.no_loc) ctx ~pname (ps : prior_spec) : Ir.prior_dist =
  (* Prefix every diagnostic message with the parameter name so users
     can locate bad priors in models with many parameters. *)
  let qualify msg = Printf.sprintf "parameter '%s': %s" pname msg in
  let err_invalid_placeholder = Ir.Uniform { Ir.lower = 0.0; Ir.upper = 1.0 } in

  (* Signature check: distribution name must be known. *)
  let expected_args = match prior_arg_signature ps.ps_name with
    | Some args -> args
    | None ->
      Diagnostics.error ctx.diags
        ~code:"E232" ~loc
        ~message:(qualify (Printf.sprintf "unknown prior distribution '%s'" ps.ps_name))
        ~detail:"Valid distributions: uniform, normal, log_normal, half_normal, beta, gamma, exponential."
        ~hint:"Check the spelling and available distributions."
        ();
      []
  in
  if expected_args = [] && prior_arg_signature ps.ps_name = None then
    err_invalid_placeholder
  else begin
    (* Signature check: duplicate kwargs. *)
    let seen = Hashtbl.create 4 in
    List.iter (fun (k, _) ->
      if Hashtbl.mem seen k then
        Diagnostics.error ctx.diags
          ~code:"E234" ~loc
          ~message:(qualify (Printf.sprintf "duplicate argument '%s' in prior '%s'" k ps.ps_name))
          ~hint:"Keyword arguments may appear at most once."
          ()
      else
        Hashtbl.add seen k ()
    ) ps.ps_args;

    (* Signature check: unknown kwargs.
       m19/C4/C10 in 2026-04-19 review: observation likelihoods use
       `normal(mean=..., sd=...)`; priors use `normal(mu=..., sigma=...)`.
       Users routinely mix them up. If the typo is one of these, the
       hint names the correct spelling explicitly. *)
    let mean_mu_hint k =
      match k, ps.ps_name with
      | ("mean", ("normal" | "log_normal")) ->
        Some "prior `normal` / `log_normal` uses `mu` (not `mean`); \
              `mean`/`sd` are used in observation likelihoods"
      | ("sd", ("normal" | "log_normal" | "half_normal")) ->
        Some "prior `normal` / `log_normal` / `half_normal` uses \
              `sigma` (not `sd`); `mean`/`sd` are used in observation \
              likelihoods"
      | _ -> None
    in
    List.iter (fun (k, _) ->
      if not (List.mem k expected_args) then
        let hint = match mean_mu_hint k with
          | Some h -> h
          | None   -> "Remove the unknown argument or check the spelling."
        in
        Diagnostics.error ctx.diags
          ~code:"E233" ~loc
          ~message:(qualify (Printf.sprintf "unknown argument '%s' for prior '%s'" k ps.ps_name))
          ~detail:(Printf.sprintf "Distribution '%s' accepts: %s." ps.ps_name (String.concat ", " expected_args))
          ~hint
          ()
    ) ps.ps_args;

    (* Resolve each expected arg to a constant float. *)
    let get_float key =
      match List.assoc_opt key ps.ps_args with
      | Some e ->
        if is_const_expr e then eval_const_expr ctx e
        else begin
          Diagnostics.error ctx.diags
            ~code:"E230" ~loc
            ~message:(qualify (Printf.sprintf "prior argument '%s' must be a compile-time constant" key))
            ~detail:(Printf.sprintf "In ~ %s(...), the argument '%s' is not a constant expression. \
                                     Prior arguments must be numeric literals, arithmetic of literals, \
                                     or pure math functions (log, exp, sqrt, ...)." ps.ps_name key)
            ~hint:"Use a numeric literal or literal arithmetic, e.g. mu = log(0.3)"
            ();
          0.0
        end
      | None ->
        Diagnostics.error ctx.diags
          ~code:"E231" ~loc
          ~message:(qualify (Printf.sprintf "prior '%s' missing required argument '%s'" ps.ps_name key))
          ~detail:(Printf.sprintf "The distribution %s requires a '%s' argument." ps.ps_name key)
          ~hint:(Printf.sprintf "Add '%s = <value>' to the prior arguments." key)
          ();
        0.0
    in
    let vals = List.map (fun k -> (k, get_float k)) expected_args in

    (* Value validation: per-distribution constraints. *)
    (match validate_prior_values ps.ps_name vals with
     | None -> ()
     | Some msg ->
       Diagnostics.error ctx.diags
         ~code:"E235" ~loc
         ~message:(qualify (Printf.sprintf "invalid prior '%s': %s" ps.ps_name msg))
         ~hint:"Check the distribution's domain: shapes/rates/sigmas must be positive, uniform lower < upper."
         ());

    let v k = List.assoc k vals in
    match ps.ps_name with
    | "uniform"     -> Ir.Uniform { Ir.lower = v "lower"; Ir.upper = v "upper" }
    | "normal"      -> Ir.Normal_p { Ir.mean = v "mu"; Ir.sd = v "sigma" }
    | "log_normal"  -> Ir.LogNormal { Ir.mu = v "mu"; Ir.sigma = v "sigma" }
    | "half_normal" -> Ir.HalfNormal { Ir.sigma = v "sigma" }
    | "beta"        -> Ir.Beta { Ir.alpha = v "alpha"; Ir.beta = v "beta" }
    | "gamma"       -> Ir.Gamma { Ir.shape = v "shape"; Ir.rate = v "rate" }
    | "exponential" -> Ir.Exponential { Ir.rate = v "rate" }
    | _ -> err_invalid_placeholder (* unreachable — name was validated above *)
  end

(** Classify a prior as plain (float-valued args) or hierarchical
    (expression-valued args, e.g. parameter references). A prior is
    hierarchical iff:
    - the declaration has an explicit `| dim` pool clause (ps_pool_over
      is Some), OR
    - any argument expression contains a non-constant term (parameter
      reference) — this allows flat-scalar leaves with hyperparent
      references, without forcing a pooling dimension.
    Wave 2 / malaria #3. *)
let classify_and_resolve_prior_spec ?(loc = Diagnostics.no_loc) ctx ~pname
      (ps : prior_spec) : prior_classification =
  let has_non_const_arg =
    List.exists (fun (_, e) -> not (is_const_expr e)) ps.ps_args
  in
  let is_hierarchical = ps.ps_pool_over <> None || has_non_const_arg in
  if not is_hierarchical then
    `Plain (resolve_prior_spec ~loc ctx ~pname ps)
  else begin
    (* Validate distribution name but allow parameter references in args. *)
    let qualify msg = Printf.sprintf "parameter '%s': %s" pname msg in
    (match prior_arg_signature ps.ps_name with
     | Some _ -> ()
     | None ->
       Diagnostics.error ctx.diags
         ~code:"E232" ~loc
         ~message:(qualify (Printf.sprintf "unknown prior distribution '%s'" ps.ps_name))
         ~detail:"Valid distributions: uniform, normal, log_normal, half_normal, beta, gamma, exponential."
         ~hint:"Check the spelling and available distributions."
         ());
    let resolved_args = List.map (fun (k, e) ->
      (k, normalize_expr (resolve_expr ctx [] e))
    ) ps.ps_args in
    (* Validate every parameter reference in the resolved args points
       at a declared parameter. Unknown names are typos or misuse. *)
    let param_names = List.filter_map (function
      | PScalar  { pname; _ } -> Some pname
      | PIndexed { pname; _ } -> Some pname
    ) ctx.param_decls in
    let rec check_refs e =
      match e with
      | Ir.Param n when not (List.mem n param_names) ->
        Diagnostics.error ctx.diags
          ~code:"E230" ~loc
          ~message:(qualify (Printf.sprintf
            "prior argument references unknown parameter '%s'" n))
          ~detail:"Hierarchical priors may reference hyperparameters \
                   declared in the same `parameters { }` block. \
                   `%s` is not a declared parameter."
          ~hint:"Check spelling, or declare the hyperparameter first."
          ()
      | Ir.Param _ | Ir.Const _ | Ir.Projected | Ir.Time | Ir.Dt -> ()
      | Ir.BinOp b -> check_refs b.left; check_refs b.right
      | Ir.UnOp  u -> check_refs u.arg
      | Ir.Cond  c -> check_refs c.pred; check_refs c.then_; check_refs c.else_
      | Ir.Pop _ | Ir.PopSum _ -> ()  (* caught elsewhere *)
      | Ir.TimeFunc _ -> ()
      | Ir.TableLookup (_, args) -> List.iter check_refs args
      | Ir.UncheckedDim u -> check_refs u.inner
    in
    List.iter (fun (_, e) -> check_refs e) resolved_args;
    `Hierarchical {
      Ir.hkind      = Ir.hierarchical_kind_of_name ps.ps_name;
      Ir.hargs      = resolved_args;
      Ir.hpool_over = Option.value ~default:"" ps.ps_pool_over;
    }
  end

let expand_parameters ctx =
  let from_params = List.concat_map (fun pd ->
    match pd with
    | PScalar { pname; pbounds; pkind; pdim; pprior; ploc } ->
      let bounds = resolve_bounds ctx pbounds in
      let pk = Some (param_kind_to_string pkind) in
      let loc = diag_loc_of_ast_ctx ctx ploc in
      let (prior, hierarchical) = match pprior with
        | None -> (None, None)
        | Some ps -> (match classify_and_resolve_prior_spec ctx ~loc ~pname ps with
                      | `Plain p        -> (Some p, None)
                      | `Hierarchical h -> (None, Some h))
      in
      [{ Ir.name          = pname;
         Ir.value         = None;
         Ir.bounds        = bounds;
         Ir.prior         = prior;
         Ir.hierarchical  = hierarchical;
         Ir.transform     = None;
         Ir.initial_value = None;
         Ir.param_kind    = pk;
         Ir.param_dim     = pdim;
       }]
    | PIndexed { pname; pdims = [dim]; pbounds; pkind; pdim; pprior; ploc } ->
      let vals = dim_values ctx dim in
      let bounds = resolve_bounds ctx pbounds in
      let pk = Some (param_kind_to_string pkind) in
      let loc = diag_loc_of_ast_ctx ctx ploc in
      let (prior, hierarchical) = match pprior with
        | None -> (None, None)
        | Some ps -> (match classify_and_resolve_prior_spec ctx ~loc ~pname ps with
                      | `Plain p        -> (Some p, None)
                      | `Hierarchical h -> (None, Some h))
      in
      List.map (fun v ->
        { Ir.name          = pname ^ "_" ^ v;
          Ir.value         = None;
          Ir.bounds        = bounds;
          Ir.prior         = prior;
          Ir.hierarchical  = hierarchical;
          Ir.transform     = None;
          Ir.initial_value = None;
          Ir.param_kind    = pk;
          Ir.param_dim     = pdim;
        }
      ) vals
    | PIndexed { pname; pdims; _ } ->
      (* The parser only produces single-dim indexed params
         (pdims = [dim]). The single-dim arm above matches that; this
         fallback is defensive. M10 in the 2026-04-19 review —
         previously this raised `failwith` which produced a bare
         stack trace in production via compile_detail_result's
         generic exn → Error catch. Even though the review's author
         identified this as "parser only produces single-dim", a
         future parser extension to multi-dim indexed params would
         regress this into a crash. Emit a real diagnostic instead. *)
      Diagnostics.error ctx.diags
        ~code:"E274"
        ~loc:Diagnostics.no_loc
        ~message:(Printf.sprintf
          "indexed parameter '%s' has %d dimensions; only single-dim \
           indexed parameters are supported"
          pname (List.length pdims))
        ~hint:"declare one parameter per stratified axis, e.g. \
               `R0[patch] : positive` rather than `R0[patch, age]`"
        ();
      []
  ) ctx.param_decls in
  (* Typed const let bindings → fixed-value parameters *)
  let from_lets = List.filter_map (fun (lb : let_binding) ->
    match lb.lkind with
    | Some pk when is_const_expr lb.lbody ->
      let v = eval_const_expr ctx lb.lbody in
      Some { Ir.name          = lb.lname;
             Ir.value         = Some v;
             Ir.bounds        = None;
             Ir.prior         = None;
             Ir.hierarchical  = None;
             Ir.transform     = None;
             Ir.initial_value = None;
             Ir.param_kind    = Some (param_kind_to_string pk);
             Ir.param_dim     = None;
           }
    | _ -> None
  ) ctx.let_bindings in
  from_params @ from_lets

(* ── Compartment expansion ───────────────────────────────────────────────── *)

let expand_compartments ctx =
  List.concat_map (fun cd ->
    let names = expand_compartment_name ctx cd.cname in
    List.map (fun name ->
      let ir_kind : Ir.compartment_kind = match cd.ckind with
        | Integer -> Ir.Integer
        | Real    -> Ir.Real
      in
      ({ Ir.name; Ir.kind = ir_kind } : Ir.compartment)
    ) names
  ) ctx.comp_decls

(* ── Table expansion ─────────────────────────────────────────────────────── *)

(** Extract a string path from the first *positional* argument of a
    function call. Only positional args are considered — previously
    (m22 in the 2026-04-19 review) this used List.find_map over all
    args regardless of keyword, so `read("file.tsv", default =
    "fallback.tsv")` could surface either string first depending on
    evaluation order. Positional-only means the path must always be
    the first arg by position, matching the documented
    `read(PATH, column = ...)` surface syntax. *)
let extract_path_arg ctx func_name args =
  let path_opt = List.find_map (fun (kw, e) ->
    if kw = "" then
      match e with EIdent (s, _) -> Some s | _ -> None
    else None
  ) args in
  (match path_opt with
   | None ->
     Diagnostics.error ctx.diags
       ~code:"E200"
       ~loc:Diagnostics.no_loc
       ~message:(Printf.sprintf
         "%s: expected a positional string path as the first argument"
         func_name)
       ~hint:"example: read(\"pop.tsv\", column = \"patch\")"
       ();
   | Some _ -> ());
  path_opt

let rec flatten_expr_list ctx (dim_entries : table_dim_entry list) = function
  | EList es     -> List.concat_map (flatten_expr_list ctx dim_entries) es
  | EConst f     -> [Ir.Const f]
  | EUnit (f, u) -> [Ir.Const (unit_to_model_time ctx f u)]
  | other        -> [resolve_expr ctx [] other]

(** Determine table source: External if `external("name")`, otherwise Inline. *)
let table_source_of_expr ctx (dim_entries : table_dim_entry list) e =
  match e with
  | EFuncCall ("external", args) ->
    (match extract_path_arg ctx "external" args with
     | None -> Ir.Inline []
     | Some name -> Ir.External name)
  | _ ->
    let vals = flatten_expr_list ctx dim_entries e in
    Ir.Inline vals

let expand_tables ctx =
  List.concat_map (fun td ->
    let dim_entries = td.tdims in
    let primary_name = List.hd td.tnames in
    let table_unit = extract_table_unit ctx ~table_name:primary_name dim_entries in
    let cell_kind = Option.map param_kind_to_string td.tcell_kind in
    match td.tvalue with
    | EFuncCall ("read", args) ->
      (* Multi-value loader: produces one Ir.table per name in td.tnames *)
      (match extract_path_arg ctx "read" args with
       | None -> []
       | Some path ->
         let default_val = match List.find_map (fun (k, e) ->
             if k = "default" then Some e else None) args with
           | Some (EConst f) -> Some f
           | _ -> None
         in
         let n_values = List.length td.tnames in
         let arrays = load_table_data ctx path
           ~dims:dim_entries ~n_values ~default_val in
         List.mapi (fun col_idx name ->
           let arr = List.nth arrays col_idx in
           let vals = Array.to_list (Array.map (fun f -> Ir.Const f) arr) in
           let vals = match table_unit with
             | Some u -> scale_table_values ctx ~table_name:name ~unit:u vals
             | None   -> vals
           in
           { Ir.name          = name;
             Ir.source        = Ir.Inline vals;
             Ir.out_of_bounds = Ir.Error;
             Ir.cell_kind     = cell_kind;
           }
         ) td.tnames)
    | _ ->
      (* Single-value path: external() or inline literal *)
      let name = match td.tnames with [n] -> n | _ ->
        Diagnostics.error ctx.diags ~code:"E215" ~loc:Diagnostics.no_loc
          ~message:"multi-name table declaration requires read(...)" ();
        List.hd td.tnames
      in
      let source = table_source_of_expr ctx dim_entries td.tvalue in
      let source = match source, table_unit with
        | Ir.Inline vs, Some u ->
          Ir.Inline (scale_table_values ctx ~table_name:name ~unit:u vs)
        | _ -> source
      in
      (match source with
       | Ir.Inline [] -> []   (* empty inline = compile error upstream, skip *)
       | _ -> [{ Ir.name; Ir.source; Ir.out_of_bounds = Ir.Error;
                 Ir.cell_kind = cell_kind }])
  ) ctx.table_decls

(* ── Initial conditions ──────────────────────────────────────────────────── *)

let is_all_const e =
  let rec walk = function
    | Ir.Const _ -> true
    | Ir.BinOp b -> walk b.left && walk b.right
    | Ir.UnOp u  -> walk u.arg
    | _           -> false
  in walk e

let eval_const ctx e =
  (* M14 in the 2026-04-19 review: before this, the UnOp arm was
     missing here but present in `is_all_const`, so `init { S = -5 }`
     produced `Ir.UnOp { Neg, Const 5.0 }`, passed the all_const
     check, and then fell into the catch-all here, emitting a
     false E402 and silently setting the init to 0.0. Same for
     floor/ceil/abs/exp/log/sqrt of constants. Fix: mirror
     autodiff's `simplify` by evaluating each UnOp arm directly. *)
  let rec eval = function
    | Ir.Const f -> f
    | Ir.BinOp { op = Ir.Add; left; right } -> eval left +. eval right
    | Ir.BinOp { op = Ir.Sub; left; right } -> eval left -. eval right
    | Ir.BinOp { op = Ir.Mul; left; right } -> eval left *. eval right
    | Ir.BinOp { op = Ir.Div; left; right } -> eval left /. eval right
    | Ir.BinOp { op = Ir.Pow; left; right } -> eval left ** eval right
    | Ir.UnOp  { op = Ir.Neg;   arg } -> -. (eval arg)
    | Ir.UnOp  { op = Ir.Exp;   arg } -> exp (eval arg)
    | Ir.UnOp  { op = Ir.Log;   arg } -> log (eval arg)
    | Ir.UnOp  { op = Ir.Sqrt;  arg } -> sqrt (eval arg)
    | Ir.UnOp  { op = Ir.Abs;   arg } -> abs_float (eval arg)
    | Ir.UnOp  { op = Ir.Floor; arg } -> floor (eval arg)
    | Ir.UnOp  { op = Ir.Ceil;  arg } -> ceil (eval arg)
    | Ir.UnOp  { op = Ir.Sin;   arg } -> sin (eval arg)
    | Ir.UnOp  { op = Ir.Cos;   arg } -> cos (eval arg)
    | Ir.UnOp  { op = Ir.Tanh;  arg } -> tanh (eval arg)
    | _ ->
      Diagnostics.error ctx.diags ~code:"E402" ~loc:Diagnostics.no_loc
        ~message:"initial condition value is not a constant expression"
        ~hint:"Use numeric literals or arithmetic of constants for init values."
        ();
      0.0
  in eval e

let expand_init ctx =
  (* Hashtbl + queue to implement override-by-source-order: later entries win,
     but insertion order is preserved for deterministic output. *)
  let tbl   : (string, Ir.expr) Hashtbl.t = Hashtbl.create 64 in
  let order : string Queue.t = Queue.create () in
  let add_entry name value =
    if not (Hashtbl.mem tbl name) then Queue.add name order;
    Hashtbl.replace tbl name value
  in
  List.iter (fun ie ->
    if ie.ibindings = [] then begin
      (* Positional or bare form *)
      let concrete_name =
        if ie.iindices = [] then ie.icomp
        else
          let idx_vals = List.map (function
            | IPosn (EIdent (s, _))     -> s
            | IPosn (EConst f)          -> string_of_float f
            | INamed (_, EIdent (s, _)) -> s
            | _                         -> "?"
          ) ie.iindices in
          String.concat "_" (ie.icomp :: idx_vals)
      in
      let resolved = normalize_expr (resolve_expr ctx [] ie.ivalue) in
      add_entry concrete_name resolved
    end else begin
      (* Loop binding form *)
      let combos = cartesian_product ie.ibindings ctx in
      List.iter (fun env ->
        let parts = name_parts_from_bindings ie.ibindings env in
        let concrete_name =
          if parts = [] then ie.icomp
          else ie.icomp ^ "_" ^ String.concat "_" parts
        in
        let resolved = normalize_expr (resolve_expr ctx env ie.ivalue) in
        add_entry concrete_name resolved
      ) combos
    end
  ) ctx.init_entries;
  let entries = Queue.fold (fun acc name ->
    acc @ [(name, Hashtbl.find tbl name)]
  ) [] order in
  if List.for_all (fun (_, e) -> is_all_const e) entries then
    Ir.Explicit (List.map (fun (k, e) -> (k, eval_const ctx e)) entries)
  else
    Ir.Parameterized entries

(* ── Simulate / output ───────────────────────────────────────────────────── *)

let expand_simulate ctx =
  match ctx.simulate with
  | None ->
    { Ir.t_start = 0.0; Ir.t_end = 100.0;
      Ir.time_semantics = "continuous"; Ir.dt = None; Ir.rng_seed = None }
  | Some sd ->
    let t_start = resolve_float_expr ctx sd.sim_from in
    let t_end   = resolve_float_expr ctx sd.sim_to   in
    { Ir.t_start; Ir.t_end;
      Ir.time_semantics = "continuous"; Ir.dt = None; Ir.rng_seed = None }

let expand_output ctx =
  let t_end = match ctx.simulate with
    | None    -> 100.0
    | Some sd -> resolve_float_expr ctx sd.sim_to
  in
  let step = match ctx.output_decl with
    | Some od -> (match od.out_trajectories with
      | Some ot -> resolve_float_expr ctx ot.otevery
      | None    -> 1.0)
    | None    -> 1.0
  in
  let format = match ctx.output_decl with
    | Some od -> (match od.out_trajectories with
      | Some ot -> ot.otformat
      | None    -> "tsv")
    | None    -> "tsv"
  in
  { Ir.times        = Ir.OutRegular { Ir.start = 0.0; Ir.step = step; Ir.end_ = t_end };
    Ir.format       = format;
    Ir.trajectory   = true;
    Ir.observations = true;
  }

(* ── Intervention expansion ──────────────────────────────────────────────── *)

(** Resolve an AST expression to a bare compartment name. Used for
    `from =` / `to =` kwargs in transfer actions.

    Previously returned `"?"` silently when the expression resolved
    to anything other than `Ir.Pop` (C7 in the 2026-04-19 review).
    The resulting intervention had `src: "?"` / `dst: "?"` — a
    compartment name that doesn't exist — and downstream consumers
    happily carried the garbage. Fix: emit E264 naming the kind of
    expression we actually got, returning "?" as the continuation
    so any other errors in the same intervention surface too. *)
let resolve_comp_name ctx env e =
  match resolve_expr ctx env e with
  | Ir.Pop name -> name
  | other ->
    let kind = match other with
      | Ir.Param p    -> Printf.sprintf "parameter reference ('%s')" p
      | Ir.PopSum _   -> "a sum of populations (PopSum)"
      | Ir.BinOp _    -> "an arithmetic expression"
      | Ir.UnOp _     -> "a unary expression"
      | Ir.Const _    -> "a constant"
      | Ir.Cond _     -> "a conditional"
      | Ir.TimeFunc _ -> "a time-function reference"
      | Ir.TableLookup _ -> "a table lookup"
      | Ir.Time       -> "the time symbol"
      | Ir.Dt         -> "the integrator step `dt`"
      | Ir.Projected  -> "a projected value"
      | Ir.Pop _      -> "a compartment" (* unreachable by pattern *)
      | Ir.UncheckedDim _ -> "a dimensional-escape expression"
    in
    Diagnostics.error ctx.diags
      ~code:"E264"
      ~loc:Diagnostics.no_loc
      ~message:(Printf.sprintf
        "expected a bare compartment name, got %s" kind)
      ~hint:"`from =` / `to =` in a transfer action must name a \
             compartment directly (e.g. `from = S`, not `from = S + R`)"
      ();
    "?"

(* ── Time function expansion ──────────────────────────────────────────────── *)

(** Load times and values for one level of an indexed interpolated function.
    Reads the file, finds columns by name from header, filters rows where the
    key column equals key_val. Returns (times, values) as float lists. *)
let load_interpolated_for_level ctx path ~key_col ~key_val ~time_col ~value_col =
  let key_ci   = ref (-1) in
  let time_ci  = ref 0 in
  let value_ci = ref 0 in
  let times  = ref [] in
  let values = ref [] in
  let on_header headers =
    let find_col name =
      match List.find_index (fun h -> h = name) headers with
      | Some i -> i
      | None ->
        Diagnostics.error ctx.diags ~code:"E219" ~loc:Diagnostics.no_loc
          ~message:(Printf.sprintf "%s: column '%s' not found in header" path name) ();
        0
    in
    key_ci   := if key_col = "" then -1 else find_col key_col;
    time_ci  := find_col time_col;
    value_ci := find_col value_col
  in
  let on_row _row_num cols =
    let get i = String.trim (try List.nth cols i with _ -> "") in
    if !key_ci < 0 || get !key_ci = key_val then begin
      (match float_of_string_opt (get !time_ci) with
       | Some t -> times  := t :: !times
       | None   -> ());
      (match float_of_string_opt (get !value_ci) with
       | Some v -> values := v :: !values
       | None   -> ())
    end
  in
  let on_done () = (List.rev !times, List.rev !values) in
  match read_csv_rows ctx path ~on_header ~on_row ~on_done with
  | Some result -> result
  | None -> ([], [])

(** Resolve a func_decl kwarg to an Ir.expr, preserving Param references.
    Emits a diagnostic and returns Const 0.0 if the key is missing. *)
let get_expr_kwarg ctx kwargs key =
  match List.assoc_opt key kwargs with
  | None   ->
    Diagnostics.error ctx.diags ~code:"E403" ~loc:Diagnostics.no_loc
      ~message:(Printf.sprintf "time function missing required argument '%s'" key)
      ~hint:(Printf.sprintf "Add '%s = <value>' to the forcing function body." key)
      ();
    Ir.Const 0.0
  | Some e -> resolve_expr ctx [] e

let get_expr_list_kwarg ctx kwargs key =
  match List.assoc_opt key kwargs with
  | None   ->
    Diagnostics.error ctx.diags ~code:"E403" ~loc:Diagnostics.no_loc
      ~message:(Printf.sprintf "time function missing required argument '%s'" key)
      ~hint:(Printf.sprintf "Add '%s = <value>' to the forcing function body." key)
      ();
    []
  | Some e -> match e with
    | EList es -> List.map (resolve_expr ctx []) es
    | _ -> [resolve_expr ctx [] e]

let expand_time_function_one ctx fname (env : (string * string) list) fkind (funit : unit_lit) fargs =
  let get_kw key =
    match List.assoc_opt key fargs with
    | None   ->
      Diagnostics.error ctx.diags ~code:"E403" ~loc:Diagnostics.no_loc
        ~message:(Printf.sprintf "time function '%s' missing required argument '%s'" fname key)
        ~hint:(Printf.sprintf "Add '%s = <value>' to the forcing function body." key)
        ();
      Ir.Const 0.0
    | Some e -> resolve_expr ctx env e
  in
  let get_kw_list key =
    match List.assoc_opt key fargs with
    | None   ->
      Diagnostics.error ctx.diags ~code:"E403" ~loc:Diagnostics.no_loc
        ~message:(Printf.sprintf "time function '%s' missing required argument '%s'" fname key)
        ~hint:(Printf.sprintf "Add '%s = <value>' to the forcing function body." key)
        ();
      []
    | Some e -> match e with
      | EList es -> List.map (resolve_expr ctx env) es
      | _ -> [resolve_expr ctx env e]
  in
  let get_str_kw key default = match List.assoc_opt key fargs with
    | Some (EIdent (s, _)) -> s
    | Some _ | None -> default
  in
  let kind = match fkind with
    | "sinusoidal" ->
      Ir.Sinusoidal {
        amplitude = get_kw "amplitude";
        period    = get_kw "period";
        phase     = get_kw "phase";
        baseline  = get_kw "baseline";
      }
    | "piecewise" ->
      Ir.Piecewise {
        breakpoints = get_kw_list "breakpoints";
        values      = get_kw_list "values";
      }
    | "interpolated" ->
      let method_ = get_str_kw "method" "linear" in
      (* File-backed form: data = "path" key_col = X time_col = Y value_col = Z *)
      (match List.assoc_opt "data" fargs with
       | Some (EIdent (path, _)) ->
         let time_col  = get_str_kw "time_col"  "time"  in
         let value_col = get_str_kw "value_col" "value" in
         (* For indexed functions, filter rows by key_col = level.
            For non-indexed functions (env is empty), read all rows. *)
         let (times, values) =
           if env = [] then
             (* Non-indexed: read all rows, no key filtering *)
             load_interpolated_for_level ctx path
               ~key_col:"" ~key_val:"" ~time_col ~value_col
           else
             let key_col = get_str_kw "key_col" "key" in
             let key_val = match List.assoc_opt key_col env with
               | Some v -> v
               | None   -> ""
             in
             load_interpolated_for_level ctx path
               ~key_col ~key_val ~time_col ~value_col
         in
         Ir.Interpolated {
           times   = List.map (fun f -> Ir.Const f) times;
           values  = List.map (fun f -> Ir.Const f) values;
           method_;
         }
       | _ ->
         Ir.Interpolated {
           times   = get_kw_list "times";
           values  = get_kw_list "values";
           method_;
         })
    | "periodic" ->
      let period_expr = get_kw "period" in
      let values =
        match List.assoc_opt "on" fargs with
        | Some on_expr ->
          (* Range-based periodic: on = [7:100, 115:199, ...]
             step = bin width (required with on).
             Generates a binary values array: 1.0 for bins in ranges, 0.0 otherwise. *)
          let step_expr = match List.assoc_opt "step" fargs with
            | Some e -> resolve_expr ctx env e
            | None ->
              Diagnostics.error ctx.diags ~code:"E404" ~loc:Diagnostics.no_loc
                ~message:(Printf.sprintf "periodic time function '%s' with 'on' requires 'step' (bin width)" fname)
                ~hint:"Add 'step = <number>' to specify the bin width for range-based periodic forcing."
                ();
              Ir.Const 1.0
          in
          let unwrap_const = function
            | Ir.Const f -> Some f
            | Ir.UncheckedDim { inner = Ir.Const f; _ } -> Some f
            | _ -> None
          in
          let period_f = match unwrap_const period_expr with Some f -> f | None ->
            Diagnostics.error ctx.diags ~code:"E405" ~loc:Diagnostics.no_loc
              ~message:(Printf.sprintf "periodic time function '%s': 'period' must be a constant when using 'on'" fname)
              ~hint:"Use a numeric literal for 'period', e.g. period = 365"
              ();
            1.0 in
          let step_f = match unwrap_const step_expr with Some f -> f | None ->
            Diagnostics.error ctx.diags ~code:"E405" ~loc:Diagnostics.no_loc
              ~message:(Printf.sprintf "periodic time function '%s': 'step' must be a constant when using 'on'" fname)
              ~hint:"Use a numeric literal for 'step', e.g. step = 1"
              ();
            1.0 in
          let n_bins = (period_f /. step_f +. 0.5) |> int_of_float in
          let arr = Array.make n_bins 0.0 in
          (* Extract ranges from the on = [...] expression *)
          let ranges = match on_expr with
            | EList items -> items
            | _ ->
              Diagnostics.error ctx.diags ~code:"E406" ~loc:Diagnostics.no_loc
                ~message:(Printf.sprintf "periodic time function '%s': 'on' must be a list of ranges" fname)
                ~hint:"Use on = [lo:hi, lo:hi, ...] to specify active ranges."
                ();
              []
          in
          List.iter (fun range ->
            match range with
            | ERange (lo_e, hi_e) ->
              let lo = match lo_e with EConst f -> int_of_float f
                | EUnit (f, u) -> int_of_float (unit_to_model_time ctx f u)
                | _ ->
                  Diagnostics.error ctx.diags ~code:"E407" ~loc:Diagnostics.no_loc
                    ~message:(Printf.sprintf "periodic time function '%s': range lower bound must be a constant" fname)
                    ~hint:"Use a numeric literal, e.g. 7:100"
                    ();
                  0 in
              let hi = match hi_e with EConst f -> int_of_float f
                | EUnit (f, u) -> int_of_float (unit_to_model_time ctx f u)
                | _ ->
                  Diagnostics.error ctx.diags ~code:"E407" ~loc:Diagnostics.no_loc
                    ~message:(Printf.sprintf "periodic time function '%s': range upper bound must be a constant" fname)
                    ~hint:"Use a numeric literal, e.g. 7:100"
                    ();
                  0 in
              let step_int = int_of_float step_f in
              if step_int > 1 && (lo mod step_int <> 0 || (hi + 1) mod step_int <> 0) then
                Diagnostics.warning ctx.diags ~code:"W301" ~loc:Diagnostics.no_loc
                  ~message:(Printf.sprintf
                    "periodic range %d:%d is not aligned to step size %d; \
                     school fraction may differ from intended value"
                    lo hi step_int)
                  ~hint:"use step = 1 for exact boundaries, or adjust ranges to multiples of step"
                  ();
              for i = lo to (min hi (n_bins - 1)) do
                arr.(i) <- 1.0
              done
            | _ ->
              Diagnostics.error ctx.diags ~code:"E406" ~loc:Diagnostics.no_loc
                ~message:(Printf.sprintf "periodic time function '%s': 'on' elements must be ranges (lo:hi)" fname)
                ~hint:"Each element of the 'on' list must be a range, e.g. on = [7:100, 115:199]"
                ()
          ) ranges;
          Array.to_list arr |> List.map (fun f -> Ir.Const f)
        | None ->
          (* Traditional form: explicit values array *)
          get_kw_list "values"
      in
      Ir.Periodic { period = period_expr; values }
    | "fourier" ->
      (* gh#59: harmonics = [(a_1, b_1), (a_2, b_2), ...].
         Accepts a list-of-pairs encoded as `[(a, b), ...]` in the
         DSL — the parser produces EList of E2-tuples / lists. *)
      let period_expr = get_kw "period" in
      let harm_arg = match List.assoc_opt "harmonics" fargs with
        | Some e -> e
        | None ->
          Diagnostics.error ctx.diags ~code:"E408" ~loc:Diagnostics.no_loc
            ~message:(Printf.sprintf "fourier forcing '%s' requires 'harmonics'" fname)
            ~hint:"Add 'harmonics = [(a1, b1), (a2, b2), ...]'"
            ();
          EList []
      in
      let pair_of = function
        | EList [a; b] -> (resolve_expr ctx env a, resolve_expr ctx env b)
        | _ ->
          Diagnostics.error ctx.diags ~code:"E408" ~loc:Diagnostics.no_loc
            ~message:(Printf.sprintf "fourier '%s': each harmonic must be a 2-element list [a, b]" fname)
            ~hint:"Use harmonics = [(a1, b1), (a2, b2), ...]"
            ();
          (Ir.Const 0.0, Ir.Const 0.0)
      in
      let harmonics = match harm_arg with
        | EList items -> List.map pair_of items
        | _ ->
          Diagnostics.error ctx.diags ~code:"E408" ~loc:Diagnostics.no_loc
            ~message:(Printf.sprintf "fourier '%s': 'harmonics' must be a list of pairs" fname)
            ~hint:"Use harmonics = [(a1, b1), (a2, b2), ...]"
            ();
          []
      in
      Ir.Fourier { period = period_expr; harmonics }
    | "periodic_spline" ->
      (* gh#59 v2 (2026-05-12): uniform knots — user gives n_basis count
         and optional degree (default 3); the evaluator derives knot
         positions from period/n_basis. See proposal at
         docs/dev/proposals/2026-05-12-periodic-bspline-algorithm.md. *)
      let period_expr = get_kw "period" in
      let unwrap_int_const expr label default =
        let ir = resolve_expr ctx env expr in
        let unwrap = function
          | Ir.Const f -> Some f
          | Ir.UncheckedDim { inner = Ir.Const f; _ } -> Some f
          | _ -> None
        in
        match unwrap ir with
        | Some f when Float.is_integer f && f >= 0.0 -> int_of_float f
        | _ ->
          Diagnostics.error ctx.diags ~code:"E408" ~loc:Diagnostics.no_loc
            ~message:(Printf.sprintf
              "periodic_spline '%s': '%s' must be a non-negative integer constant"
              fname label)
            ~hint:"Use a numeric integer literal, e.g. n_basis = 6"
            ();
          default
      in
      let n_basis = match List.assoc_opt "n_basis" fargs with
        | None ->
          Diagnostics.error ctx.diags ~code:"E408" ~loc:Diagnostics.no_loc
            ~message:(Printf.sprintf "periodic_spline '%s' missing 'n_basis'" fname)
            ~hint:"Add 'n_basis = <int>' (number of basis functions, e.g. 6)"
            ();
          1
        | Some e -> unwrap_int_const e "n_basis" 1
      in
      let degree = match List.assoc_opt "degree" fargs with
        | None -> 3
        | Some e -> unwrap_int_const e "degree" 3
      in
      let coefs = get_kw_list "coefs" in
      if n_basis <= degree then
        Diagnostics.error ctx.diags ~code:"E408" ~loc:Diagnostics.no_loc
          ~message:(Printf.sprintf
            "periodic_spline '%s': n_basis (%d) must be greater than degree (%d)"
            fname n_basis degree)
          ~hint:"Increase n_basis or lower degree."
          ();
      if List.length coefs <> n_basis then
        Diagnostics.error ctx.diags ~code:"E408" ~loc:Diagnostics.no_loc
          ~message:(Printf.sprintf
            "periodic_spline '%s': coefs has length %d but n_basis = %d"
            fname (List.length coefs) n_basis)
          ~hint:"Provide exactly n_basis coefficient values."
          ();
      Ir.PeriodicSpline { period = period_expr; n_basis; degree; coefs }
    | k ->
      Diagnostics.error ctx.diags ~code:"E408" ~loc:Diagnostics.no_loc
        ~message:(Printf.sprintf "unknown time function kind '%s' in '%s'" k fname)
        ~detail:"Supported kinds: sinusoidal, piecewise, interpolated, periodic, fourier, periodic_spline."
        ~hint:(Printf.sprintf "Change the kind to one of: sinusoidal, piecewise, interpolated, periodic, fourier, periodic_spline.")
        ();
      Ir.Piecewise { breakpoints = []; values = [] }
  in
  (* GH #8: the forcing's tier-3 unit literal drives both the stored-
     value scale normalisation and the declared dimension. Scale is 1.0
     for `'count` / `'ratio` (counts and dimensionless multipliers pass
     through); non-trivial for rate units (e.g. `'per_year` with
     `time_unit = 'days` gives 1/365.2425). The dim-checker reads
     `time_function.dim` authoritatively — no value-based inference. *)
  let scale = unit_to_model_time ctx 1.0 funit in
  let scale_expr e =
    if scale = 1.0 then e
    else Ir.BinOp { op = Mul; left = e; right = Ir.Const scale }
  in
  let kind = if scale = 1.0 then kind else
    match kind with
    | Ir.Sinusoidal s ->
      Ir.Sinusoidal { s with
        Ir.amplitude = scale_expr s.amplitude;
        Ir.baseline  = scale_expr s.baseline }
    | Ir.Piecewise p ->
      Ir.Piecewise { p with Ir.values = List.map scale_expr p.values }
    | Ir.Interpolated i ->
      Ir.Interpolated { i with Ir.values = List.map scale_expr i.values }
    | Ir.Periodic p ->
      Ir.Periodic { p with Ir.values = List.map scale_expr p.values }
    | Ir.Fourier _ ->
      (* gh#59: fourier harmonics are dimensionless modulators; scale
         applies to the declared forcing dim, not the coef values. *)
      kind
    | Ir.PeriodicSpline ps ->
      (* gh#59 v2: scale applies to the coefficient values (n_basis/
         degree are integers, period is dimensional and unchanged). *)
      Ir.PeriodicSpline { ps with Ir.coefs = List.map scale_expr ps.coefs }
  in
  let dim = unit_lit_to_dim funit in
  { Ir.name = fname; Ir.kind; Ir.dim }

(** Expand ODE equations from the DSL's `ode { X = expr }` blocks into
    IR `ode_equation` records.

    The DSL surface currently takes a bare compartment name (no
    indices); each `ode_decl` maps 1:1 to an `Ir.ode_equation`. If the
    parser is later extended with stratified ODEs, this needs a
    cartesian-product loop like `expand_time_functions`. Reported as
    C5 in the 2026-04-19 compiler review — previously
    `Ir.ode_equations` was hardcoded to `[]`, so every `ode {}` block
    was silently dropped and any `: real` compartment that depended on
    its ODE stayed frozen at its init value. Post-expansion integrity
    (`Validate.validate`, M1 in the same review) will error when a
    `Real` compartment has no emitted equation. *)
let expand_ode_equations ctx : Ir.ode_equation list =
  List.map (fun (od : ode_decl) ->
    let deriv = normalize_expr (resolve_expr ctx [] od.oderiv) in
    { Ir.compartment = od.ocomp; Ir.derivative = deriv }
  ) ctx.ode_decls

let expand_time_functions ctx : Ir.time_function list =
  List.concat_map (fun (fd : func_decl) ->
    if fd.findices = [] then
      [expand_time_function_one ctx fd.fname [] fd.fkind fd.funit fd.fargs]
    else begin
      let combos = cartesian_product fd.findices ctx in
      List.map (fun env ->
        let parts = name_parts_from_bindings fd.findices env in
        let fname = fd.fname ^ "_" ^ String.concat "_" parts in
        expand_time_function_one ctx fname env fd.fkind fd.funit fd.fargs
      ) combos
    end
  ) ctx.func_decls

let expand_scheduled_actions ctx decls ~always_active =
  let t_start = match ctx.simulate with
    | None    -> 0.0
    | Some sd -> resolve_float_expr ctx sd.sim_from
  in
  let t_end = match ctx.simulate with
    | None    -> 100.0
    | Some sd -> resolve_float_expr ctx sd.sim_to
  in
  List.concat_map (fun iv ->
    let iv_loc = diag_loc_of_ast_ctx ctx iv.ivloc in
    let base_name = if iv.ivindices = [] then None else Some iv.ivname in
    let combos = cartesian_product iv.ivindices ctx in
    List.filter_map (fun env ->
      let pass_guard = match iv.ivguard with
        | None   -> true
        | Some g -> eval_guard env g
      in
      if not pass_guard then None
      else
      let parts = name_parts_from_bindings iv.ivindices env in
      let iv_name =
        if parts = [] then iv.ivname
        else iv.ivname ^ "_" ^ String.concat "_" parts
      in
      let schedule = match iv.ivschedule with
        | SAtTimes exprs ->
          Ir.AtTimes (List.map (fun e ->
            let ir = normalize_expr (resolve_expr ctx env e) in
            match ir with Ir.Const f -> f | _ -> 0.0
          ) exprs)
        | SRecurring (every, from_opt, until_opt) ->
          let period = resolve_float_expr ctx every in
          let start  = match from_opt with
            | Some e -> resolve_float_expr ctx e
            | None   -> t_start
          in
          let end_   = match until_opt with
            | Some e -> resolve_float_expr ctx e
            | None   -> t_end
          in
          if period <= 0.0 then
            Diagnostics.error ctx.diags
              ~code:"E240" ~loc:iv_loc
              ~message:(Printf.sprintf "intervention '%s': 'every' must be positive (got %g)" iv.ivname period)
              ~hint:"Use a positive interval, e.g. every = 30 'days"
              ();
          if start > end_ then
            Diagnostics.error ctx.diags
              ~code:"E241" ~loc:iv_loc
              ~message:(Printf.sprintf "intervention '%s': 'from' (%g) must be <= 'until' (%g)" iv.ivname start end_)
              ~hint:"Either reorder the values or check unit conversions (e.g. years → days)."
              ();
          (* Cap expanded schedule length to catch accidental year-at-minute schedules. *)
          let max_fires = 1_000_000 in
          if period > 0.0 && start <= end_ then begin
            let n_fires = int_of_float (((end_ -. start) /. period) +. 1.0) in
            if n_fires > max_fires then
              Diagnostics.error ctx.diags
                ~code:"E242" ~loc:iv_loc
                ~message:(Printf.sprintf "intervention '%s' schedule expands to %d firings (cap %d)"
                            iv.ivname n_fires max_fires)
                ~hint:"Check units: e.g. every = 1 'days with until = 100 'years is 36_525 entries."
                ()
          end;
          Ir.Recurring { Ir.start; Ir.period; Ir.end_; Ir.at_day = None }
        | SEveryAtDay (every, day) ->
          let period = resolve_float_expr ctx every in
          let at_day = resolve_float_expr ctx day in
          Ir.Recurring { Ir.start = t_start; Ir.period; Ir.end_ = t_end; Ir.at_day = Some at_day }
      in
      let actions = match iv.ivaction with
        | ATransfer kwargs ->
          (* Validate the kwarg shape first (C6 in the 2026-04-19
             review). Before this, a missing/typoed `fraction` or
             `count` silently produced `actions = []` — the
             intervention fired on schedule and did nothing. A missing
             `from`/`to` produced `src = "?"` / `dst = "?"` which the
             emitted IR happily carried as a non-existent compartment
             reference. All silent-wrong-answer class. *)
          let has_from     = List.mem_assoc "from"     kwargs in
          let has_to       = List.mem_assoc "to"       kwargs in
          let has_fraction = List.mem_assoc "fraction" kwargs in
          let has_count    = List.mem_assoc "count"    kwargs in
          let known = ["from"; "to"; "fraction"; "count"] in
          let unknown = List.filter_map (fun (k, _) ->
            if k = "" || List.mem k known then None else Some k) kwargs in
          let err code msg hint =
            Diagnostics.error ctx.diags ~code ~loc:Diagnostics.no_loc
              ~message:msg ~hint ()
          in
          if not has_from then
            err "E261" (Printf.sprintf
              "intervention '%s': transfer action missing `from =`" iv.ivname)
              "example: transfer(from = S, to = V, fraction = 0.8)";
          if not has_to then
            err "E261" (Printf.sprintf
              "intervention '%s': transfer action missing `to =`" iv.ivname)
              "example: transfer(from = S, to = V, fraction = 0.8)";
          if not (has_fraction || has_count) then
            err "E261" (Printf.sprintf
              "intervention '%s': transfer action needs either \
               `fraction =` or `count =`" iv.ivname)
              "fraction = 0.0..1.0 (relative) OR count = N (absolute)";
          if has_fraction && has_count then
            err "E261" (Printf.sprintf
              "intervention '%s': transfer action has both `fraction` \
               and `count` — these are mutually exclusive" iv.ivname)
              "pick one: fraction for a proportion, count for an \
               absolute number";
          List.iter (fun k ->
            err "E262" (Printf.sprintf
              "intervention '%s': unknown transfer kwarg '%s'"
              iv.ivname k)
              "valid kwargs: from, to, fraction, count"
          ) unknown;
          let src = match List.assoc_opt "from" kwargs with
            | Some e -> resolve_comp_name ctx env e
            | None   -> "?"
          in
          let dst = match List.assoc_opt "to" kwargs with
            | Some e -> resolve_comp_name ctx env e
            | None   -> "?"
          in
          (match List.assoc_opt "fraction" kwargs with
          | Some fe ->
            [Ir.FractionTransfer { Ir.src; Ir.dst; Ir.fraction = resolve_expr ctx env fe }]
          | None ->
            match List.assoc_opt "count" kwargs with
            | Some ce ->
              [Ir.AbsoluteTransfer { Ir.src; Ir.dst; Ir.count = resolve_expr ctx env ce }]
            | None -> [])
        | ASet (comp, idxs, expr) ->
          let idx_vals = List.map (index_item_to_str env) idxs in
          let concrete = if idx_vals = [] then comp
            else String.concat "_" (comp :: idx_vals) in
          (* m13 in 2026-04-19 review: the parser's iv_kv catch-all
             produces `ASet(unknown_key, [], expr)` for any
             `foo = expr` inside an intervention body, so a typo like
             `fraction` vs `fracton` silently becomes an action on a
             non-existent compartment. Flag here with a specific
             E-code instead of letting it pass through and surface
             downstream as a generic E503 unknown-compartment. *)
          if not (Hashtbl.mem ctx.expanded_comp_tbl concrete
                  || Hashtbl.mem ctx.comp_tbl comp) then
            Diagnostics.error ctx.diags
              ~code:"E265"
              ~loc:iv_loc
              ~message:(Printf.sprintf
                "intervention '%s' sets '%s' which is not a declared \
                 compartment"
                iv_name concrete)
              ~hint:"check the compartments block, or fix the kwarg name \
                     (e.g. fraction, count, from, to)"
              ();
          [Ir.Set { Ir.compartment = concrete; Ir.value = resolve_expr ctx env expr }]
        | AAdd (comp, idxs, expr) ->
          let idx_vals = List.map (index_item_to_str env) idxs in
          let concrete = if idx_vals = [] then comp
            else String.concat "_" (comp :: idx_vals) in
          [Ir.AddAction { Ir.add_compartment = concrete; Ir.add_count = resolve_expr ctx env expr }]
      in
      Some { Ir.name = iv_name; Ir.base_name; Ir.schedule; Ir.actions;
             Ir.always_active = always_active }
    ) combos
  ) decls

let expand_interventions ctx =
  expand_scheduled_actions ctx ctx.interv_decls ~always_active:false
  @ expand_scheduled_actions ctx ctx.event_decls ~always_active:true

(* ── Observation model expansion ─────────────────────────────────────────── *)

let expand_observations ctx =
  List.concat_map (fun od ->
    (* m25 in 2026-04-19 review: indexed obs with no explicit
       data_stream defaults each stratum to its own stream name
       (afp_cases_kano, afp_cases_borno, ...) per IR spec §5.3.
       This is a common UX trap — users often want a single multi-
       column file keyed by the base name. Warn once per declaration. *)
    let od_loc = diag_loc_of_ast_ctx ctx od.oloc in
    (match od.odata_stream with
     | None when od.oindices <> [] ->
       Diagnostics.warning ctx.diags
         ~code:"W203"
         ~loc:od_loc
         ~message:(Printf.sprintf
           "indexed observation '%s' has no explicit data_stream; \
            each stratum gets its own stream name"
           od.oname)
         ~hint:"add `data_stream = \"<name>\"` to share one stream, \
                or set it explicitly per stratum to silence this warning"
         ()
     | _ -> ());
    (* m12 in 2026-04-19 review: each of schedule / projection /
       likelihood is required. Previously the parser filled in
       Poisson(rate=1) / every=1 / incidence(name) defaults, so an
       empty block compiled to a silently-meaningless likelihood. *)
    let missing_field name =
      Diagnostics.error ctx.diags
        ~code:"E266"
        ~loc:od_loc
        ~message:(Printf.sprintf
          "observation '%s': missing required field '%s'" od.oname name)
        ~hint:(match name with
          | "schedule" -> "add `every = <period>` or `at = [t1, t2, ...]`"
          | "projection" -> "add `incidence = <transition>` or `prevalence = <compartment>`"
          | "likelihood" -> "add `likelihood = poisson(rate = ...)`, `neg_binomial(mean = ..., r = ...)`, etc."
          | _ -> "required field")
        ()
    in
    let sched_v = match od.oschedule with
      | Some s -> s
      | None -> missing_field "schedule"; ObsEvery (EConst 1.0)
    in
    let proj_v = match od.oprojection with
      | Some p -> p
      | None -> missing_field "projection"; ProjIncidence (od.oname, [])
    in
    let lik_v = match od.olikelihood with
      | Some l -> l
      | None -> missing_field "likelihood"; LikPoisson [("rate", EConst 1.0)]
    in
    let combos = cartesian_product od.oindices ctx in
    (* If no indices, combos = [[]] — one iteration with empty env *)
    List.filter_map (fun env ->
    let t_start = match ctx.simulate with
      | None    -> 0.0
      | Some sd -> resolve_float_expr ctx sd.sim_from
    in
    let t_end = match ctx.simulate with
      | None    -> 100.0
      | Some sd -> resolve_float_expr ctx sd.sim_to
    in
    let schedule = match sched_v with
      | ObsEvery every ->
        let step = resolve_float_expr ctx every in
        Ir.ObsRegular { Ir.start = t_start; Ir.step; Ir.end_ = t_end }
      | ObsTimes ts ->
        Ir.ObsAtTimes (List.map (resolve_float_expr ctx) ts)
    in
    (* `prevalence(X)` projects a compartment snapshot at observation time.
       If X is Erlang- or otherwise-stratified, the bare name has no concrete
       expansion — the user means "sum over all strata," matching how the
       same bare name in a rate expression expands to PopSum (see
       `resolve_ident_name`, §5.1 of the language spec). Emit CurrentPopSum
       when the base name is a declared compartment with >1 expansions. *)
    let prevalence_projection base idx_vals =
      let concrete = if idx_vals = [] then base
        else String.concat "_" (base :: idx_vals) in
      if Hashtbl.mem ctx.expanded_comp_tbl concrete then
        Ir.CurrentPop concrete
      else if idx_vals = [] && Hashtbl.mem ctx.comp_tbl base then
        (* Bare stratified compartment — sum over all strata. *)
        let expansions = expand_compartment_name ctx base in
        (match expansions with
         | [single] -> Ir.CurrentPop single
         | many     -> Ir.CurrentPopSum many)
      else
        Ir.CurrentPop concrete  (* Unknown — let the Rust side emit a clean diagnostic. *)
    in
    let projection = match proj_v with
      | ProjIncidence (name, idxs) ->
        let idx_vals = List.map (index_item_to_str env) idxs in
        let concrete = if idx_vals = [] then name
          else String.concat "_" (name :: idx_vals) in
        Ir.CumulativeFlow concrete
      | ProjPrevalence (name, idxs) ->
        let idx_vals = List.map (index_item_to_str env) idxs in
        prevalence_projection name idx_vals
      | ProjDerived (EFuncCall ("incidence", args)) ->
        (match List.assoc_opt "" args with
         | Some (EIdent (n, _))    -> Ir.CumulativeFlow n
         | Some (EIndex (n, idxs)) ->
           Ir.CumulativeFlow (String.concat "_" (n :: List.map (index_item_to_str env) idxs))
         | _ -> Ir.CumulativeFlow "?")
      | ProjDerived (EFuncCall ("prevalence", args)) ->
        (match List.assoc_opt "" args with
         | Some (EIdent (n, _))    -> prevalence_projection n []
         | Some (EIndex (n, idxs)) ->
           prevalence_projection n (List.map (index_item_to_str env) idxs)
         | _ -> Ir.CurrentPop "?")
      | ProjDerived (EIdent (name, _)) ->
        (* Disambiguate: is this a compartment (prevalence) or transition (flow)? *)
        if Hashtbl.mem ctx.expanded_comp_tbl name then
          Ir.CurrentPop name
        else if Hashtbl.mem ctx.comp_tbl name then
          prevalence_projection name []
        else
          Ir.CumulativeFlow name
      | ProjDerived (EIndex (name, idxs)) ->
        let idx_vals = List.map (index_item_to_str env) idxs in
        let concrete = String.concat "_" (name :: idx_vals) in
        if Hashtbl.mem ctx.expanded_comp_tbl concrete then
          Ir.CurrentPop concrete
        else if Hashtbl.mem ctx.comp_tbl name then
          prevalence_projection name idx_vals
        else
          Ir.CumulativeFlow concrete
      | ProjDerived e ->
        Ir.DerivedExpr (resolve_expr ctx env e)
    in
    (* Likelihood kwarg resolution with strict diagnostics. Unlike the
       silent 0.0 default of old, we emit a real error for:
         E250 — missing required kwarg (or only positional args supplied)
         E251 — unknown kwarg name (typo / wrong distribution)
       Mirrors E231/E233 on priors. *)
    let lik_name = match lik_v with
      | LikNegBinomial _  -> "neg_binomial"
      | LikPoisson _      -> "poisson"
      | LikNormal _       -> "normal"
      | LikBinomial _     -> "binomial"
      | LikBetaBinomial _ -> "beta_binomial"
      | LikBernoulli _    -> "bernoulli"
    in
    let required_kwargs = match lik_v with
      | LikNegBinomial _  -> ["mean"; "r"]
      | LikPoisson _      -> ["rate"]
      | LikNormal _       -> ["mean"; "sd"]
      | LikBinomial _     -> ["n"; "p"]
      | LikBetaBinomial _ -> ["n"; "alpha"; "beta"]
      | LikBernoulli _    -> ["p"]
    in
    let current_kwargs = match lik_v with
      | LikNegBinomial k | LikPoisson k | LikNormal k
      | LikBinomial k | LikBetaBinomial k | LikBernoulli k -> k
    in
    (* Report unknown kwargs and positional args up front. *)
    List.iter (fun (k, _) ->
      if k = "" then
        Diagnostics.error ctx.diags
          ~code:"E250" ~loc:od_loc
          ~message:(Printf.sprintf
            "observation '%s': likelihood '%s' requires named arguments \
             (got a positional argument)" od.oname lik_name)
          ~hint:(Printf.sprintf "Use '%s' — e.g. %s(%s = ...)"
            (String.concat " = ..., " required_kwargs)
            lik_name
            (List.hd required_kwargs))
          ()
      else if not (List.mem k required_kwargs) then
        Diagnostics.error ctx.diags
          ~code:"E251" ~loc:od_loc
          ~message:(Printf.sprintf
            "observation '%s': likelihood '%s' has no argument '%s'"
            od.oname lik_name k)
          ~hint:(Printf.sprintf "Expected: %s"
            (String.concat ", " required_kwargs))
          ()
    ) current_kwargs;
    let resolve_kw kwargs name =
      match List.assoc_opt name kwargs with
      | Some e -> resolve_expr ctx env e
      | None   ->
        Diagnostics.error ctx.diags
          ~code:"E250" ~loc:Diagnostics.no_loc
          ~message:(Printf.sprintf
            "observation '%s': likelihood '%s' missing required argument '%s'"
            od.oname lik_name name)
          ~hint:(Printf.sprintf "Add '%s = <expr>' to the likelihood — e.g. %s(%s = projected)"
            name lik_name name)
          ();
        Ir.Const 0.0
    in
    let likelihood = match lik_v with
      | LikNegBinomial kwargs ->
        Ir.NegBinomial {
          Ir.mean       = resolve_kw kwargs "mean";
          Ir.dispersion = resolve_kw kwargs "r";
        }
      | LikPoisson kwargs ->
        Ir.Poisson { Ir.rate = resolve_kw kwargs "rate" }
      | LikNormal kwargs ->
        Ir.Normal {
          Ir.mean = resolve_kw kwargs "mean";
          Ir.sd   = resolve_kw kwargs "sd";
        }
      | LikBinomial kwargs ->
        Ir.Binomial {
          Ir.n = resolve_kw kwargs "n";
          Ir.p = resolve_kw kwargs "p";
        }
      | LikBetaBinomial kwargs ->
        Ir.BetaBinomial {
          Ir.n     = resolve_kw kwargs "n";
          Ir.alpha = resolve_kw kwargs "alpha";
          Ir.beta  = resolve_kw kwargs "beta";
        }
      | LikBernoulli kwargs ->
        Ir.Bernoulli { Ir.p = resolve_kw kwargs "p" }
    in
    let parts = name_parts_from_bindings od.oindices env in
    let obs_name =
      if parts = [] then od.oname
      else od.oname ^ "_" ^ String.concat "_" parts
    in
    let data_stream = Option.value ~default:obs_name od.odata_stream in
    Some { Ir.name        = obs_name;
      Ir.data_stream;
      Ir.schedule;
      Ir.projection;
      Ir.likelihood;
    }
    ) combos
  ) ctx.obs_decls

(* ── Hierarchical-prior cycle / self-reference check ─────────────────────── *)

(** Collect the set of parameter names referenced anywhere inside an
    AST expression. Used by the cycle detector. *)
let rec collect_param_refs known_params acc = function
  | EConst _ | EUnit _ -> acc
  | EIdent (name, _) when List.mem name known_params -> name :: acc
  | EIdent (_, _) -> acc
  | EIndex (_, items) ->
    List.fold_left (fun a item ->
      match item with
      | IPosn e | INamed (_, e) -> collect_param_refs known_params a e
    ) acc items
  | EBinOp (_, l, r) ->
    let a = collect_param_refs known_params acc l in
    collect_param_refs known_params a r
  | EUnOp (_, e) -> collect_param_refs known_params acc e
  | ESum (_, _, body) -> collect_param_refs known_params acc body
  | ECond (p, t, e) ->
    let a = collect_param_refs known_params acc p in
    let a = collect_param_refs known_params a t in
    collect_param_refs known_params a e
  | EFuncCall (_, args) ->
    List.fold_left (fun a (_, e) -> collect_param_refs known_params a e) acc args
  | EList es ->
    List.fold_left (fun a e -> collect_param_refs known_params a e) acc es
  | ERange (lo, hi) ->
    let a = collect_param_refs known_params acc lo in
    collect_param_refs known_params a hi

(** Check hierarchical prior reference graph for self-references and
    cycles. Wave 2 / malaria #3 Gate 2 — risks C1, C2. Legitimate deep
    chains (risk C3) pass cleanly. *)
let check_hierarchical_cycles ctx =
  let known_params = List.filter_map (function
    | PScalar  { pname; _ } -> Some pname
    | PIndexed { pname; _ } -> Some pname
  ) ctx.param_decls in
  (* Build adjacency: param → list of params its prior references. *)
  let adj = Hashtbl.create 16 in
  List.iter (fun pd ->
    let (pname, pprior) = match pd with
      | PScalar  { pname; pprior; _ } -> (pname, pprior)
      | PIndexed { pname; pprior; _ } -> (pname, pprior)
    in
    match pprior with
    | None -> Hashtbl.replace adj pname []
    | Some ps ->
      let refs = List.fold_left (fun acc (_, e) ->
        collect_param_refs known_params acc e
      ) [] ps.ps_args in
      Hashtbl.replace adj pname (List.sort_uniq compare refs)
  ) ctx.param_decls;

  (* DFS-based cycle detection. Emits E236 with a clear message. *)
  let visited  = Hashtbl.create 16 in
  let on_stack = Hashtbl.create 16 in
  let rec dfs node path =
    if Hashtbl.mem on_stack node then begin
      (* Cycle detected — path contains the cycle. *)
      let cycle_nodes =
        let rec take_from acc = function
          | [] -> List.rev acc
          | n :: _ when n = node -> List.rev (n :: acc)
          | n :: rest -> take_from (n :: acc) rest
        in take_from [] path
      in
      let desc =
        if List.length cycle_nodes <= 1 then
          Printf.sprintf "parameter '%s' references itself in its prior" node
        else
          Printf.sprintf "cycle in hierarchical prior references: %s -> %s"
            (String.concat " -> " cycle_nodes) node
      in
      Diagnostics.error ctx.diags
        ~code:"E236"
        ~loc:Diagnostics.no_loc
        ~message:desc
        ~hint:"hierarchical priors must form a DAG: hyperparents declared \
               independently, leaves reference them"
        ()
    end
    else if not (Hashtbl.mem visited node) then begin
      Hashtbl.add on_stack node ();
      let neighbours = try Hashtbl.find adj node with Not_found -> [] in
      List.iter (fun n -> dfs n (node :: path)) neighbours;
      Hashtbl.remove on_stack node;
      Hashtbl.add visited node ()
    end
  in
  Hashtbl.iter (fun node _ -> dfs node []) adj

(* ── Shadowing check ──────────────────────────────────────────────────────── *)

(** Emit W103 for any let binding whose name also appears as a stratum value. *)
let check_shadowing ctx =
  let all_strat_vals = List.concat_map (fun sd ->
    let vs = match List.assoc_opt sd.sdim ctx.dim_registry with Some vs -> vs | None -> [] in
    List.map (fun v -> (v, sd.sdim)) vs
  ) ctx.stratifies in
  List.iter (fun lb ->
    match List.assoc_opt lb.lname all_strat_vals with
    | None -> ()
    | Some dim ->
      Diagnostics.warning ctx.diags
        ~code:"W103"
        ~loc:Diagnostics.no_loc
        ~message:(Printf.sprintf
          "let binding '%s' shadows stratum value '%s' in dimension '%s'. \
           This is allowed but consider renaming."
          lb.lname lb.lname dim)
        ()
  ) ctx.let_bindings

(* ── Scenarios expansion ─────────────────────────────────────────────────── *)

(** Resolved, pre-IR scenario form. Built in two passes: first collect
    this scenario's own fields into a ResolvedScen, then fold parent
    chain through `extends`. Expressions in set/scale remain unresolved
    here so that children can reference parent-resolved values. *)
type resolved_scen = {
  rs_name    : string;
  rs_label   : string option;             (* None = use rs_name as default *)
  rs_enable  : string list;
  rs_disable : string list;
  rs_set     : (string * expr) list;     (* still exprs — resolved after merge *)
  rs_scale   : (string * expr) list;
  rs_compose : string list;
  rs_t_end   : expr option;               (* still expr — resolved after merge *)
  rs_parent  : string option;
}

(** Closest-by-edit-distance scenario name suggestion for an unknown
    parent. Returns [None] if nothing is within 3 edits. *)
let suggest_scenario_name (candidates : string list) (target : string) : string option =
  let edit_distance a b =
    let la = String.length a and lb = String.length b in
    let m = Array.make_matrix (la + 1) (lb + 1) 0 in
    for i = 0 to la do m.(i).(0) <- i done;
    for j = 0 to lb do m.(0).(j) <- j done;
    for i = 1 to la do
      for j = 1 to lb do
        let cost = if a.[i-1] = b.[j-1] then 0 else 1 in
        m.(i).(j) <- min (min (m.(i-1).(j) + 1) (m.(i).(j-1) + 1)) (m.(i-1).(j-1) + cost)
      done
    done;
    m.(la).(lb)
  in
  candidates
  |> List.map (fun c -> (c, edit_distance target c))
  |> List.sort (fun (_, a) (_, b) -> compare a b)
  |> List.filter (fun (_, d) -> d <= 3)
  |> (function (name, _) :: _ -> Some name | [] -> None)

(** Merge parent fields under the child's. For each field, apply the
    rule documented in the plan:
    - label / t_end: child overrides parent (if child specified)
    - set / scale: child keys override parent keys on collision; union otherwise
    - enable / disable / compose: append parent first, then child, dedup
      while preserving first-occurrence order. Emits a loud info! log
      when this actually changes the resolved list vs the child's own
      list (surfaces the footgun).
    Does not resolve expressions — that happens in the final pass. *)
let merge_fields ctx ~child ~parent ~parent_name =
  (* Append-and-dedup: parent first, then child; keep first occurrence. *)
  let dedup_concat parent_list child_list =
    let seen = Hashtbl.create 4 in
    let combined = parent_list @ child_list in
    List.filter (fun x ->
      if Hashtbl.mem seen x then false
      else (Hashtbl.add seen x (); true)
    ) combined
  in
  let merged_enable  = dedup_concat parent.rs_enable  child.rs_enable  in
  let merged_disable = dedup_concat parent.rs_disable child.rs_disable in
  let merged_compose = dedup_concat parent.rs_compose child.rs_compose in
  (* Loud log when the append changed things: child-only enables did NOT
     capture the full picture. Only fires when the parent contributed
     something beyond the child's own list. *)
  (* Loud warning (Diagnostics has no Info level) when the append-dedup
     actually changed the resolved list — surfaces the footgun where a
     child declares `enable = [X]` intending "only X" but the parent
     contributes more entries. *)
  let changed name cl ml =
    if ml <> cl then
      Diagnostics.warning ctx.diags ~code:"W310" ~loc:Diagnostics.no_loc
        ~message:(Printf.sprintf
          "scenario '%s' inherits %s from '%s': resolved %s = [%s] \
           (child declared [%s])"
          child.rs_name name parent_name name
          (String.concat "; " ml) (String.concat "; " cl))
        ~hint:"`extends` appends parent's enable/disable/compose to the child's. \
               To remove a parent's intervention, put it in `disable`."
        ()
  in
  changed "enable"  child.rs_enable  merged_enable;
  changed "disable" child.rs_disable merged_disable;
  changed "compose" child.rs_compose merged_compose;
  {
    rs_name    = child.rs_name;
    rs_label   = (match child.rs_label with Some _ as l -> l | None -> parent.rs_label);
    rs_enable  = merged_enable;
    rs_disable = merged_disable;
    (* Keep both parent's and child's set entries in order so that the
       child's expression can reference the parent's resolved value.
       Duplicate keys are resolved by HashMap overwrite during the
       final resolution pass (later entry wins). *)
    rs_set     = parent.rs_set   @ child.rs_set;
    rs_scale   = parent.rs_scale @ child.rs_scale;
    rs_compose = merged_compose;
    rs_t_end   = (match child.rs_t_end with Some _ as t -> t | None -> parent.rs_t_end);
    rs_parent  = None;  (* post-resolve *)
  }

(** Collect a scenario_decl's own fields into a ResolvedScen, without
    resolving parent or expressions. *)
let collect_own_fields (sd : scenario_decl) : resolved_scen =
  let label    = ref None in
  let enable   = ref [] in
  let disable  = ref [] in
  let set_ps   = ref [] in
  let scale_ps = ref [] in
  let compose  = ref [] in
  let t_end    = ref None in
  let parent   = ref None in
  List.iter (function
    | ScLabel s    -> label := Some s
    | ScEnable es  -> enable := !enable @ es
    | ScDisable ds -> disable := !disable @ ds
    | ScSet ps     -> set_ps := !set_ps @ ps
    | ScScale ps   -> scale_ps := !scale_ps @ ps
    | ScCompose cs -> compose := !compose @ cs
    | ScTEnd e     -> t_end := Some e
    | ScExtends p  -> parent := Some p
  ) sd.scfields;
  { rs_name = sd.scname;
    rs_label = !label;
    rs_enable = !enable;
    rs_disable = !disable;
    rs_set = !set_ps;
    rs_scale = !scale_ps;
    rs_compose = !compose;
    rs_t_end = !t_end;
    rs_parent = !parent;
  }

(** Resolve parent chain for one scenario. DFS with visiting set for
    cycle detection (E25x) and depth counter for code-smell cap (E25z).
    Returns the fully-merged resolved_scen (expressions still unresolved). *)
let resolve_parents ctx (decl_map : (string * scenario_decl) list) (own : resolved_scen)
    : resolved_scen =
  let max_depth = 5 in
  let rec go visiting depth scen =
    match scen.rs_parent with
    | None -> scen
    | Some parent_name ->
      if List.mem parent_name visiting then begin
        let chain = (scen.rs_name :: visiting |> List.rev) @ [parent_name] in
        Diagnostics.error ctx.diags ~code:"E25x" ~loc:Diagnostics.no_loc
          ~message:(Printf.sprintf "scenario extends cycle: %s"
                      (String.concat " → " chain))
          ~hint:"remove one of the `extends` in the cycle."
          ();
        { scen with rs_parent = None }   (* stop descent after error *)
      end
      else if depth >= max_depth then begin
        Diagnostics.error ctx.diags ~code:"E25z" ~loc:Diagnostics.no_loc
          ~message:(Printf.sprintf
            "scenario '%s' extends chain exceeds %d — refactor, or submit a \
             feature request for multi-parent composition"
            scen.rs_name max_depth)
          ~hint:"Chains longer than 5 are a code smell; factor common \
                 ancestors into shared base scenarios, or combine into one \
                 scenario if they're really the same configuration."
          ();
        { scen with rs_parent = None }
      end
      else begin
        match List.assoc_opt parent_name decl_map with
        | None ->
          let all_names = List.map fst decl_map in
          let hint = match suggest_scenario_name all_names parent_name with
            | Some s -> Printf.sprintf "Did you mean '%s'?" s
            | None -> "No scenario by that name is defined in this model."
          in
          Diagnostics.error ctx.diags ~code:"E25y" ~loc:Diagnostics.no_loc
            ~message:(Printf.sprintf "scenario '%s' extends unknown scenario '%s'"
                        scen.rs_name parent_name)
            ~hint:hint
            ();
          { scen with rs_parent = None }
        | Some parent_decl ->
          let parent_own = collect_own_fields parent_decl in
          let parent_resolved = go (scen.rs_name :: visiting) (depth + 1) parent_own in
          merge_fields ctx ~child:scen ~parent:parent_resolved ~parent_name
      end
  in
  go [] 0 own

let expand_scenarios ctx : Ir.preset list =
  (* Pass 1: build name → declaration lookup. *)
  let decl_map : (string * scenario_decl) list =
    List.map (fun sd -> (sd.scname, sd)) ctx.scenario_decls
  in
  (* Pass 2: for each scenario, resolve parents then emit IR preset. *)
  List.map (fun sd ->
    let own = collect_own_fields sd in
    let resolved = resolve_parents ctx decl_map own in
    (* Expression resolution with parent-first semantics:
       parent's `set` values become bindings for the child's set
       expressions. Fold left-to-right, substituting any EIdent that
       matches a prior name with its resolved numeric value. This
       bypasses ctx.let_bindings (which is finalized at compile start)
       and keeps the substitution scoped to this scenario. *)
    let rec subst bindings expr =
      match expr with
      | EConst _ | EUnit _ -> expr
      | EIdent (n, _) when List.mem_assoc n bindings ->
        EConst (List.assoc n bindings)
      | EIdent _ -> expr
      | EUnOp (op, e) -> EUnOp (op, subst bindings e)
      | EBinOp (op, l, r) -> EBinOp (op, subst bindings l, subst bindings r)
      | EFuncCall (name, args) ->
        EFuncCall (name, List.map (fun (k, e) -> (k, subst bindings e)) args)
      | other -> other
    in
    (* Left-to-right fold with overwrite-on-duplicate. Each expression
       is substituted using every prior binding (so a child's
       `beta = beta * 1.5` reads the parent's resolved beta), then
       resolved to f64. When the same key appears twice — always the
       case when a child overrides a parent's set — the later value
       wins in the final output. First-seen order is preserved. *)
    let resolve_fold vs =
      (* m23 in 2026-04-19 review: previously this rebuilt `bindings`
         via Hashtbl.fold on every iteration, making the fold O(N²).
         Maintain bindings incrementally — `subst` only reads the
         latest value per key, which is what Hashtbl.replace already
         provides when we pass the full bindings list with most recent
         entries first. *)
      let map = Hashtbl.create (List.length vs) in
      let order = ref [] in
      let bindings = ref [] in
      List.iter (fun (k, e) ->
        let e' = subst !bindings e in
        let v = resolve_float_expr ctx e' in
        if not (Hashtbl.mem map k) then order := k :: !order;
        Hashtbl.replace map k v;
        bindings := (k, v) :: !bindings
      ) vs;
      List.rev !order |> List.map (fun k -> (k, Hashtbl.find map k))
    in
    let set_vals   = resolve_fold resolved.rs_set in
    let scale_vals = resolve_fold resolved.rs_scale in
    let t_end_val  = Option.map (resolve_float_expr ctx) resolved.rs_t_end in
    { Ir.preset_name    = resolved.rs_name;
      Ir.preset_label   = Option.value resolved.rs_label ~default:resolved.rs_name;
      Ir.preset_params  = set_vals;
      Ir.preset_enable  = resolved.rs_enable;
      Ir.preset_disable = resolved.rs_disable;
      Ir.preset_scale   = scale_vals;
      Ir.preset_compose = resolved.rs_compose;
      Ir.preset_t_end   = t_end_val;
    }
  ) ctx.scenario_decls

(* ── Top-level expand ─────────────────────────────────────────────────────── *)

(* ── Model structure ─────────────────────────────────────────────────────── *)

(** Recover the pre-expansion base name of an expanded transition name by
    prefix-matching against the known set from ctx. Relies on the compiler
    invariant that expanded names are {base}_{stratum_parts} with '_'. *)
(* Longest-prefix wins — M15 in the 2026-04-19 review. If a model
   declares both `foo` and `foo_bar`, then matches against expanded
   name `foo_bar_child`, `List.find_opt` would return whichever was
   declared first; if `foo` came first, the expanded name was
   misattributed to base `foo` (when it actually belongs to
   `foo_bar`). model_structure fields downstream
   (transmission_transitions, infectious_compartments) then carried
   wrong bases. Fix: sort candidates by base-name length descending
   before find_opt, so the longest matching prefix wins. *)
let find_base_trname ctx ename =
  List.sort (fun a b ->
    compare (String.length b.trname) (String.length a.trname)) ctx.transitions
  |> List.find_opt (fun td ->
    let b = td.trname and bl = String.length td.trname and el = String.length ename in
    ename = b || (el > bl && String.sub ename 0 bl = b && ename.[bl] = '_')
  )
  |> Option.map (fun td -> td.trname)

(** Same invariant: compartment expanded names are {base}_{dim_values}.
    Same longest-prefix-wins fix as find_base_trname above. *)
let find_base_compname ctx expanded_name =
  List.sort (fun a b ->
    compare (String.length b.cname) (String.length a.cname)) ctx.comp_decls
  |> List.find_opt (fun cd ->
    let b = cd.cname and bl = String.length cd.cname and el = String.length expanded_name in
    expanded_name = b || (el > bl && String.sub expanded_name 0 bl = b && expanded_name.[bl] = '_')
  )
  |> Option.map (fun cd -> cd.cname)

let build_model_structure ctx expanded_trs =
  let dimensions = List.filter_map (fun sd ->
    match List.assoc_opt sd.sdim ctx.dim_registry with
    | Some vs -> Some { Ir.dim_name = sd.sdim; Ir.dim_values = vs }
    | None    -> None
  ) ctx.stratifies in
  let base_compartments = List.map (fun cd -> cd.cname) ctx.comp_decls in
  let compartment_dims = List.map (fun cd ->
    (cd.cname, comp_dims ctx cd.cname)
  ) ctx.comp_decls in
  (* Collect Pop/PopSum names from the numerator of a rate expression.
     Descends through every subexpression that isn't strictly a
     denominator, so compartments that appear only in N = S+I+R
     (denominator) are excluded but compartments inside Sub/Min/Max/
     Pow/UnOp are included. For `beta * S * max(I - Q, 0) / N` this
     yields {S, I, Q}, not {S} — M16 in the 2026-04-19 review.
     Prior version fell through to `acc` for Sub/Min/Max/Pow/Mod/
     UnOp/TimeFunc/TableLookup, missing infectious compartments
     hidden behind any of those forms. *)
  let rec collect_numerator_pops acc = function
    | Ir.Pop n -> n :: acc
    | Ir.PopSum ns -> ns @ acc
    | Ir.BinOp { op = Ir.Div; left; _ } ->
      (* Deliberately do NOT descend into the right operand — that's
         the denominator and its pops aren't numerator contributions. *)
      collect_numerator_pops acc left
    | Ir.BinOp b ->
      collect_numerator_pops (collect_numerator_pops acc b.left) b.right
    | Ir.UnOp u -> collect_numerator_pops acc u.arg
    | Ir.Cond c ->
      collect_numerator_pops
        (collect_numerator_pops (collect_numerator_pops acc c.pred) c.then_)
        c.else_
    | Ir.TableLookup (_, args) ->
      List.fold_left collect_numerator_pops acc args
    | Ir.Const _ | Ir.Param _ | Ir.Time | Ir.Dt | Ir.Projected | Ir.TimeFunc _ -> acc
    | Ir.UncheckedDim u -> collect_numerator_pops acc u.inner
  in
  let seen_tr  = Hashtbl.create 4 in
  let seen_inf = Hashtbl.create 4 in
  let transmission_transitions = ref [] in
  let infectious_compartments  = ref [] in
  List.iter (fun (t : Ir.transition) ->
    match t.metadata with
    | Some { Ir.origin_kind = Some "transmission"; Ir.source_compartment; _ } ->
      (match find_base_trname ctx t.name with
       | Some b when not (Hashtbl.mem seen_tr b) ->
         Hashtbl.add seen_tr b ();
         transmission_transitions := b :: !transmission_transitions
       | _ -> ());
      (* Infectious compartments = pops referenced in rate that are NOT the source. *)
      let src_base = Option.bind source_compartment (find_base_compname ctx) in
      let rate_pops = collect_numerator_pops [] t.rate in
      List.iter (fun pop_name ->
        match find_base_compname ctx pop_name with
        | Some b when Some b <> src_base && not (Hashtbl.mem seen_inf b) ->
          Hashtbl.add seen_inf b ();
          infectious_compartments := b :: !infectious_compartments
        | _ -> ()
      ) rate_pops
    | _ -> ()
  ) expanded_trs;
  { Ir.dimensions;
    Ir.compartment_dims;
    Ir.base_compartments;
    Ir.transmission_transitions = List.rev !transmission_transitions;
    Ir.infectious_compartments  = List.rev !infectious_compartments;
  }

(* ── L401 lint: discretization-correction with fixed time literal ──
   Catches the AST shape `(1 - exp(-RATE * Const c))` (or with `c * RATE`).
   The user almost always meant `dt` instead of the literal — pinning to
   a specific time literal makes the rate correct only when the runtime
   `--dt` matches that literal. See docs/dev/warning-catalog.md §L401. *)

let rec expr_contains_param_or_pop = function
  | Ir.Param _ | Ir.Pop _ | Ir.PopSum _ -> true
  | Ir.BinOp { left; right; _ } ->
    expr_contains_param_or_pop left || expr_contains_param_or_pop right
  | Ir.UnOp { arg; _ } -> expr_contains_param_or_pop arg
  | Ir.Cond { pred; then_; else_ } ->
    expr_contains_param_or_pop pred
    || expr_contains_param_or_pop then_
    || expr_contains_param_or_pop else_
  | Ir.UncheckedDim u -> expr_contains_param_or_pop u.inner
  | Ir.TableLookup (_, args) -> List.exists expr_contains_param_or_pop args
  | Ir.Const _ | Ir.Time | Ir.Dt | Ir.Projected | Ir.TimeFunc _ -> false

(* Treat `UncheckedDim { inner = Const c; ... }` (the IR form of unit
   literals like `1 'days`) as a constant for L401 matching. *)
let as_const = function
  | Ir.Const c -> Some c
  | Ir.UncheckedDim { inner = Ir.Const c; _ } -> Some c
  | _ -> None

(* Strip an outer UnOp Neg if present, returning the inner expr. *)
let strip_neg = function
  | Ir.UnOp { op = Ir.Neg; arg } -> arg
  | e -> e

(* Detect `exp(arg)` where arg = -(RATE * Const) under some normalization:
     - UnOp Neg around a Mul of RATE and Const
     - Mul where one operand is `-RATE` (Neg around param-bearing) and the
       other is a Const (or UncheckedDim Const).
   RATE must contain a Param or Pop (i.e. not be purely constant). *)
let exp_arg_matches_neg_rate_times_const arg =
  let try_mul l r =
    (* Try: l is rate-bearing (possibly negated) and r is constant. *)
    let l_inner = strip_neg l in
    let r_inner = strip_neg r in
    match as_const l_inner, expr_contains_param_or_pop r_inner,
          as_const r_inner, expr_contains_param_or_pop l_inner with
    | Some c, true, _, _ -> Some c
    | _, _, Some c, true -> Some c
    | _ -> None
  in
  match arg with
  | Ir.UnOp { op = Ir.Neg; arg = Ir.BinOp { op = Ir.Mul; left; right } } ->
    try_mul left right
  | Ir.BinOp { op = Ir.Mul; left; right } ->
    (* Match `(-RATE) * Const` or `Const * (-RATE)` (Neg absorbed). *)
    let neg_left =
      match left with Ir.UnOp { op = Ir.Neg; _ } -> true | _ -> false in
    let neg_right =
      match right with Ir.UnOp { op = Ir.Neg; _ } -> true | _ -> false in
    if neg_left || neg_right then try_mul left right else None
  | _ -> None

(* Detect the canonical `1 - exp(-RATE * Const)` shape at this node. *)
let detect_l401_at_node = function
  | Ir.BinOp { op = Ir.Sub;
               left = Ir.Const c;
               right = Ir.UnOp { op = Ir.Exp; arg } }
    when c = 1.0 -> exp_arg_matches_neg_rate_times_const arg
  | _ -> None

let rec walk_expr_for_l401 ~on_match e =
  (match detect_l401_at_node e with
   | Some lit -> on_match lit
   | None -> ());
  match e with
  | Ir.BinOp { left; right; _ } ->
    walk_expr_for_l401 ~on_match left;
    walk_expr_for_l401 ~on_match right
  | Ir.UnOp { arg; _ } -> walk_expr_for_l401 ~on_match arg
  | Ir.Cond { pred; then_; else_ } ->
    walk_expr_for_l401 ~on_match pred;
    walk_expr_for_l401 ~on_match then_;
    walk_expr_for_l401 ~on_match else_
  | Ir.UncheckedDim u -> walk_expr_for_l401 ~on_match u.inner
  | Ir.TableLookup (_, args) ->
    List.iter (walk_expr_for_l401 ~on_match) args
  | Ir.Const _ | Ir.Param _ | Ir.Pop _ | Ir.PopSum _
  | Ir.Time | Ir.Dt | Ir.Projected | Ir.TimeFunc _ -> ()

let lint_l401 ctx (expanded_trs : Ir.transition list) =
  (* Avoid duplicate emits when the same expanded transition rate fires
     the pattern more than once (e.g. β-correction used twice in a sum). *)
  let seen = Hashtbl.create 4 in
  List.iter (fun (t : Ir.transition) ->
    walk_expr_for_l401 t.rate ~on_match:(fun _lit ->
      if not (Hashtbl.mem seen t.name) then begin
        Hashtbl.add seen t.name ();
        Diagnostics.warning ctx.diags ~code:"L401" ~loc:Diagnostics.no_loc
          ~message:(Printf.sprintf
            "transition '%s' rate uses a fixed time literal inside an Euler-correction \
             pattern `(1 - exp(-RATE * <literal>))` — likely meant `dt`"
            t.name)
          ~hint:("the canonical discretization-correct form is "
                 ^ "`(1 - exp(-RATE * dt)) / dt`; pinning to a literal "
                 ^ "makes the rate correct only when the runtime --dt "
                 ^ "matches that literal. See docs/dev/warning-catalog.md §L401.")
          ()
      end)
  ) expanded_trs

(** Compute the identity-tracked subgraph (2026-05-19 proposal,
    §"Identity-tracked subgraph (inferred)").

    1. Seed: destinations of `#[lineage]` events ∪ parent-pool
       compartments of `#[lineage]` events.
    2. Close under: for every transition c1 → c2, if c1 is tracked, add c2.
    3. Result: every compartment whose individuals should carry IDs.

    Forward reachability over the transition graph, closed under cycles
    (SIRS R→S waning: once R is tracked, R→S pulls S in; the worklist
    fixpoint terminates because the tracked set is bounded by the
    compartment set). Returns the tracked compartments in stable
    (compartment-declaration) order. Empty when there are no `#[lineage]`
    transitions — the lineage subsystem is then statically inert. *)
let compute_identity_tracked
    (compartments : Ir.compartment list)
    (transitions  : Ir.transition list) : string list =
  let module SS = Set.Make (String) in
  (* Seed from lineage events: destinations + parent pools. *)
  let seed =
    List.fold_left (fun acc (t : Ir.transition) ->
      match t.Ir.lineage with
      | None -> acc
      | Some l ->
        let acc =
          match t.Ir.metadata with
          | Some { Ir.dest_compartment = Some d; _ } -> SS.add d acc
          | _ -> acc
        in
        List.fold_left (fun acc (comp, _) -> SS.add comp acc)
          acc l.Ir.parent_pool_weights
    ) SS.empty transitions
  in
  (* Forward closure: c1 → c2 edges from transition metadata. Iterate to
     a fixpoint (handles cycles without infinite recursion). *)
  let edges =
    List.filter_map (fun (t : Ir.transition) ->
      match t.Ir.metadata with
      | Some { Ir.source_compartment = Some s; Ir.dest_compartment = Some d; _ } ->
        Some (s, d)
      | _ -> None
    ) transitions
  in
  let rec close tracked =
    let tracked' =
      List.fold_left (fun acc (s, d) ->
        if SS.mem s acc then SS.add d acc else acc
      ) tracked edges
    in
    if SS.equal tracked tracked' then tracked else close tracked'
  in
  let tracked = close seed in
  (* Stable order: follow compartment declaration order. *)
  List.filter_map (fun (c : Ir.compartment) ->
    if SS.mem c.Ir.name tracked then Some c.Ir.name else None
  ) compartments

let expand_detail ?(source_dir = "") ?(filename = "<input>") (name : string) (decls : declaration list)
    : Ir.model * context * model_summary =
  let ctx = empty_context ~source_dir ~filename () in
  collect_declarations ctx decls;
  (* Pass 1: resolve dimensions {} block, build dim_registry *)
  resolve_dimensions ctx;
  (* Build O(1) lookup tables for resolve_expr *)
  build_lookup_tables ctx;
  (* W103 shadowing check: let bindings vs stratum values *)
  check_shadowing ctx;
  (* E236: hierarchical-prior cycle / self-reference detection (#3 gate 2) *)
  check_hierarchical_cycles ctx;
  (* E217: check that guard expressions only reference dim levels / loop vars *)
  check_guards ctx;
  (* Save original transitions before desugaring *)
  ctx.orig_transitions <- ctx.transitions;
  let expanded_comps = expand_compartments ctx in
  let (expanded_trs, filtered_n) = expand_transitions_counted ctx in
  lint_l401 ctx expanded_trs;
  let ms = build_model_structure ctx expanded_trs in
  let model = {
    Ir.name               = name;
    Ir.version            = "0.3";
    Ir.time_unit          = unit_lit_to_string ctx.time_unit;
    Ir.description        = ctx.description;
    Ir.origin             = ctx.origin;
    Ir.compartments       = expanded_comps;
    Ir.transitions        = expanded_trs;
    Ir.ode_equations      = expand_ode_equations ctx;
    Ir.time_functions     = expand_time_functions ctx;
    Ir.tables             = expand_tables ctx;
    Ir.interventions      = expand_interventions ctx;
    Ir.observations       = expand_observations ctx;
    Ir.parameters         = expand_parameters ctx;
    Ir.initial_conditions = expand_init ctx;
    Ir.output             = expand_output ctx;
    Ir.simulation         = expand_simulate ctx;
    Ir.presets            = expand_scenarios ctx;
    Ir.model_structure    = Some ms;
    Ir.balance            = (match ctx.balance_decl with
      | None -> None
      | Some bd -> Some {
          Ir.balance_target = bd.bcomp;
          Ir.balance_expr   = resolve_expr ctx [] bd.bexpr;
        });
    Ir.identity_tracked_compartments =
      compute_identity_tracked expanded_comps expanded_trs;
  } in
  let summary = {
    base_compartment_count     = List.length ctx.comp_decls;
    expanded_compartment_count = List.length expanded_comps;
    base_transition_count      = List.length ctx.orig_transitions;
    expanded_transition_count  = List.length expanded_trs;
    filtered_transition_count  = filtered_n;
    let_binding_count          = List.length ctx.let_bindings;
    table_count                = List.length ctx.table_decls;
    param_count                = List.length ctx.param_decls;
    obs_count                  = List.length ctx.obs_decls;
    interv_count               = List.length ctx.interv_decls;
  } in
  (model, ctx, summary)

let expand ?(source_dir = "") ?(filename = "<input>") (name : string) (decls : declaration list) : Ir.model =
  let (model, _, _) = expand_detail ~source_dir ~filename name decls in
  model
