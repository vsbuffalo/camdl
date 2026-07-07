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
  mutable reactive_decls  : reactive_decl list;   (* gh#204 *)
  mutable quantity_decls  : quantity_decl list;   (* proposal 2026-06-25 *)
  mutable contrast_decls  : contrast_decl list;   (* counterfactual contrasts *)
  mutable diags           : Diagnostics.t;  (* collected errors/warnings *)
  mutable reads           : (string * string) list;
  (* (as-written, resolved) external data files opened during expansion, in
     reverse (most-recent-first) order. Accumulated at the single read
     chokepoint [read_csv_rows]; surfaced by [reads] and powers
     `camdlc --emit-deps`. Never affects the IR. *)
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
  (* Fix B (shared-binding extraction). A `let` whose body is state-only
     (no parameter, no other let — so d/dp ≡ 0, matching the
     `differentiate(BindingRef)=0` autodiff arm) and context-independent
     (every index-position variable is bound by the let's own declared
     indices or an enclosing `sum`) is hoisted ONCE into `model.bindings`
     and referenced by `Ir.BindingRef` instead of inlined at every use
     site. This is what shrinks the spatial FOI from O(P²A²) IR to O(P²+PA). *)
  mutable hoist_memo      : (string, bool) Hashtbl.t;   (* let name -> hoistable? (memoized) *)
  mutable hoisted_tbl     : (string, unit) Hashtbl.t;   (* concrete binding name -> already registered *)
  mutable hoisted_rev     : (string * Ir.expr) list;    (* registered bindings, reverse-topological (deps last) *)
  (* While resolving a #[lineage] transition's rate, suppress extraction so the
     lineage parent-decomposition (Lineage.classify_parents) sees the fully
     inlined rate exactly as before — it cannot see through a state-bearing
     BindingRef. The same let is still hoisted at its non-lineage use sites. *)
  mutable suppress_hoist  : bool;
  (* Declared value-column names of the observation stream currently being
     expanded (2026-06-10 observation data-entry §3). While set, an identifier
     in a likelihood expression matching one of these names resolves to
     [Ir.ObsColumnRef name] — a per-observation aux data reference the Rust
     binder fills by name (e.g. binomial `n = tested`) — instead of falling
     through name resolution to E100. Empty outside the likelihood-resolution
     scope. *)
  mutable obs_aux_cols    : string list;
  (* Resolved constant tables, indexed by name → (row-major flattened cells,
     ordered dimension names). Populated once by [build_table_index] right
     after [expand_tables] runs (and before transition expansion), so a
     compile-time `where` predicate (`sum(... where dist[p,q] < r, ...)`) can
     read a table's resolved values during [resolve_expr]. Tables resolve to
     compile-time constants, so this is the natural place for them to live —
     the same way [dim_registry] holds resolved dimensions. External
     (`--table`) tables are absent here (no compile-time values). *)
  mutable table_index    : (string, Ir.expr array * string list) Hashtbl.t;
}

(* Stratification-name predicates. Expanded names are [base] itself or a leaf
   [base_<dim values>], so recovering the base of an expanded name is a prefix
   test. [is_expansion_of] recognises the base and its leaves (used to map an
   expanded name back to its pre-expansion declaration); [is_indexed_leaf] is
   the strict variant — a genuine leaf only, excluding the base and the bare
   [base_] — used to match an indexed parameter declaration's expansions. *)
let is_expansion_of ~base name =
  let bl = String.length base in
  name = base ||
  (String.length name > bl && String.sub name 0 bl = base && name.[bl] = '_')

let is_indexed_leaf ~base name =
  let bl = String.length base in
  String.length name > bl + 1 && String.sub name 0 bl = base && name.[bl] = '_'

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
  reactive_decls       = [];
  quantity_decls       = [];
  contrast_decls       = [];
  diags                = Diagnostics.create ();
  reads                = [];
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
  hoist_memo           = Hashtbl.create 16;
  hoisted_tbl          = Hashtbl.create 64;
  hoisted_rev          = [];
  suppress_hoist       = false;
  obs_aux_cols         = [];
  table_index          = Hashtbl.create 16;
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

(* `is_leap_year` / `days_in_month` are defined here (ahead of
   `parse_iso_date`) so the date-literal parser can range-check the day
   leap-aware. They mirror `rust/crates/ir/src/caltime.rs`:
   `is_leap` / `days_in_month` use identical formulas. *)
let is_leap_year y =
  (y mod 4 = 0 && y mod 100 <> 0) || y mod 400 = 0

let days_in_month y m =
  match m with
  | 1 | 3 | 5 | 7 | 8 | 10 | 12 -> 31
  | 4 | 6 | 9 | 11              -> 30
  | 2                            -> if is_leap_year y then 29 else 28
  | _ -> invalid_arg (Printf.sprintf "days_in_month: month %d out of range" m)

(** Parse an ISO calendar date `YYYY-MM-DD` into `(year, month, day)`.

    Mirrors the canonical grammar in `rust/crates/ir/src/caltime.rs`
    `parse_iso_date` (gh#98, C6): the two parsers MUST accept exactly the
    same set of strings, or a `date()` literal that compiles on the OCaml
    side produces a different internal time than the Rust runtime computes.
    Concretely:
      - leading/trailing whitespace is trimmed,
      - a trailing zone designator (`Z`, `+HH:MM`, `-HH:MM`) is accepted
        and discarded (a bare date denotes a civil-calendar day,
        zone-independent — proposal §6.8),
      - month is validated in 1..12 and day in 1..days_in_month(y, m)
        (leap-aware), so `date("2020-02-30")` and `date("2020-13-01")` are
        rejected rather than silently shifting to a garbage day offset.

    Returns [Error msg] (a human-readable reason) rather than raising, so
    callers emit a located diagnostic (E223) instead of a [failwith] stack
    trace or a silently-absorbed `0.0`. *)
let parse_iso_date (raw : string) : (int * int * int, string) result =
  let s = String.trim raw in
  (* The date portion is the first 10 chars: YYYY-MM-DD. *)
  if String.length s < 10 then
    Error (Printf.sprintf "date '%s' is not in YYYY-MM-DD form" raw)
  else
    let date_part = String.sub s 0 10 in
    let rest = String.sub s 10 (String.length s - 10) in
    (* Classify the remainder: empty or a bare zone designator → discard;
       anything else (a `T`/space time-of-day, or junk) → reject. *)
    let is_zone =
      rest = "" || rest = "Z" || rest = "z" ||
      ((String.length rest = 6)
       && (rest.[0] = '+' || rest.[0] = '-')
       && rest.[3] = ':'
       && (let ok = ref true in
           String.iteri (fun i c ->
             if (i >= 1 && i <= 2) || (i >= 4 && i <= 5) then
               (if not (c >= '0' && c <= '9') then ok := false)) rest;
           !ok))
    in
    if not is_zone then
      Error (Printf.sprintf
        "date '%s' carries an unsupported trailer (time-of-day is not \
         supported; use a bare YYYY-MM-DD or a zone designator)" raw)
    else begin
      let digits a b =
        let ok = ref true in
        for i = a to b do
          let c = date_part.[i] in
          if not (c >= '0' && c <= '9') then ok := false
        done; !ok
      in
      let shape_ok =
        date_part.[4] = '-' && date_part.[7] = '-'
        && digits 0 3 && digits 5 6 && digits 8 9
      in
      if not shape_ok then
        Error (Printf.sprintf "date '%s' is not in YYYY-MM-DD form" raw)
      else
        let y = int_of_string (String.sub date_part 0 4) in
        let m = int_of_string (String.sub date_part 5 2) in
        let d = int_of_string (String.sub date_part 8 2) in
        if m < 1 || m > 12 then
          Error (Printf.sprintf
            "date '%s' has month %d out of range (must be 01..12)" raw m)
        else if d < 1 || d > days_in_month y m then
          Error (Printf.sprintf
            "date '%s' has day %d out of range for %04d-%02d (must be \
             01..%02d)" raw d y m (days_in_month y m))
        else Ok (y, m, d)
    end

let parse_date_to_float origin_str date_str time_unit =
  let (oy, om, od) =
    match parse_iso_date origin_str with Ok v -> v | Error m -> failwith m in
  let (ty, tm, td) =
    match parse_iso_date date_str with Ok v -> v | Error m -> failwith m in
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

(* ── Proleptic-Gregorian calendar arithmetic ─────────────────────────────────

   Used by `add_calendar_months` / `add_calendar_years` expander
   primitives (Phase 2 of the 2026-05-22 typed-time proposal §4)
   and by `date_range` for calendar cadences. These functions are
   *constant-free* in the days-per-month sense: they do real
   (year, month, day) arithmetic and never touch the 30.4369
   average-month factor.

   (`is_leap_year` / `days_in_month` are defined above, ahead of
   `parse_iso_date`, which range-checks against them.) *)

(** `add_calendar_months (y, m, d) n` — proleptic-Gregorian
    month-stepping with month-end clamping. The algorithm from
    proposal §4:

      m' = ((m - 1 + n) mod 12) + 1
      y' = y + (m - 1 + n) div 12
      d' = min(d, days_in_month(y', m'))

    OCaml's built-in [mod] and [/] truncate toward zero; we need
    Euclidean semantics so a negative `m - 1 + n` wraps correctly
    (e.g. (m=3, n=-1): (3-1-1)=1, +1=2, fine; (m=1, n=-1):
    (1-1-1)=-1; euclid_mod (-1) 12 = 11, m' = 12, year decrements). *)
let add_calendar_months_ymd (y, m, d) n =
  let total = m - 1 + n in
  let euclid_div a b =
    let q = a / b and r = a mod b in
    if (r < 0 && b > 0) || (r > 0 && b < 0) then q - 1 else q
  in
  let euclid_mod a b =
    let r = a mod b in
    if (r < 0 && b > 0) || (r > 0 && b < 0) then r + b else r
  in
  let m' = (euclid_mod total 12) + 1 in
  let y' = y + (euclid_div total 12) in
  let d' = min d (days_in_month y' m') in
  (y', m', d')

let add_calendar_years_ymd (y, m, d) n =
  (* Calendar years: same month/day shape, year offset by n, with
     clamping if the target month is Feb 29 in a non-leap year. *)
  let y' = y + n in
  let d' = min d (days_in_month y' m) in
  (y', m, d')

(** Format a (y, m, d) triple as an ISO date string. Width-padded
    to YYYY-MM-DD; years 0..9999 only (anything else is out of
    scope for camdl). *)
let format_iso_date (y, m, d) =
  Printf.sprintf "%04d-%02d-%02d" y m d

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
let read_csv_rows ctx path ~ref_desc ~ref_hint_example ~on_header ~on_row ~on_done =
  (* W104: an absolute file path is non-portable — it bakes one machine's
     filesystem layout into the model, breaking sharing, model-repo reuse, and
     `camdl mre` bundling (gh#211, gh#307). This is the shared chokepoint for
     every compile-time file read — `read(...)` tables/dimensions AND a forcing's
     `data =` time series — so a single check covers them all; [ref_desc] /
     [ref_hint_example] let each call site name the construct the author actually
     wrote (so a forcing `data =` path is not misreported as `read()`). We warn
     (not error) because an absolute path still works locally, so a hard error
     would block legitimate exploratory work; `-Werror` / `--deny` (gh#56) make
     it strict for CI. The check is on the path STRING (before the file-existence
     check below), so it fires whether or not the absolute file happens to exist
     on this machine — non-portability is a property of the path, not of local
     presence. We use [Filename.is_relative] (the stdlib portable predicate); a
     `../`-escaping *relative* path is a legitimate multi-model-repo pattern and
     is NOT flagged. *)
  if not (Filename.is_relative path) then
    Diagnostics.warning ctx.diags
      ~code:"W104"
      ~loc:Diagnostics.no_loc
      ~message:(Printf.sprintf
        "%s uses an absolute path %S — non-portable model" ref_desc path)
      ~hint:(Printf.sprintf
        "use a path relative to the .camdl source file (e.g. %s) so the \
         model runs on any machine" ref_hint_example)
      ();
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
    (* Record the (as-written, resolved) pair for the read-closure depfile
       (`camdlc --emit-deps`). Only files that actually exist and get opened
       are recorded; a missing file fired E200 above. *)
    ctx.reads <- (path, abs_path) :: ctx.reads;
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
      (* gh#144: scan past any leading `#` comment block (provenance lines
         such as source URL / fetch date — the repo's data-step convention)
         and blank lines, treating the FIRST non-comment, non-blank physical
         line as the header. This mirrors the data-row skip below, which
         already drops `#` lines among data rows; the only `#` placement
         that is meaningful is a leading block, so we skip exactly that.
         [row_num] counts PHYSICAL lines so on_row diagnostics still point at
         the right line of the file even when the header was preceded by
         comments. *)
      let row_num = ref 0 in
      let is_comment_or_blank line =
        line = "" || (String.length line > 0 && line.[0] = '#')
      in
      let rec read_header () =
        let raw_line = input_line ic in   (* raises End_of_file at EOF *)
        incr row_num;
        let line = String.trim raw_line in
        if is_comment_or_blank line then read_header ()
        else List.map String.trim (split_by sep line)
      in
      try
        let header_cols = read_header () in
        on_header header_cols;
        (try while true do
          let raw_line = input_line ic in
          incr row_num;
          let line = String.trim raw_line in
          if not (is_comment_or_blank line) then begin
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
let load_table_data ctx path ~dims ~n_values ~default_val ~cell_kind =
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
    let n_header = List.length header_cols in
    (* gh#144: the header must carry at least the n_dims index columns so
       each dimension maps to a column. A shorter header (e.g. a stray
       single-column line) cannot be zipped against dim_names — the old
       code truncated header_dims and then tripped `List.combine` on the
       length mismatch. Diagnose it cleanly here and skip the
       reorder/name checks below, which assume one header column per
       dimension. *)
    if n_header < n_dims then
      Diagnostics.error ctx.diags
        ~code:"E221"
        ~loc:Diagnostics.no_loc
        ~message:(Printf.sprintf
          "%s: header has %d column(s) but table needs at least %d index column(s) for dimensions %s"
          path n_header n_dims (String.concat " × " dim_names))
        ~hint:"the first row must be a header naming the dimension columns; \
               if it is a comment, prefix it with '#'"
        ()
    else begin
      let header_dims =
        List.init n_dims (fun i -> String.trim (List.nth header_cols i))
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
          (* header_dims and dim_names are both length n_dims here, so the
             zip is total. *)
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
            (match float_of_string_opt cell with
             | Some f -> arrays.(j).(idx) <- f
             | None ->
               (* Date-valued cell: permitted only for an instant/duration
                  cell-kind table in an anchored model. The ISO date resolves
                  to internal time via origin + time_unit at compile time —
                  the same parse_date_to_float used by date() literals and the
                  --data loader (docs/dates.md, "The one rule"). A bare number
                  already matched the Some-branch above and is taken as
                  internal time directly. *)
               (match cell_kind, ctx.origin with
                | Some ("instant" | "duration"), Some origin_str ->
                  (try arrays.(j).(idx) <- parse_date_to_float origin_str cell ctx.time_unit
                   with Failure msg | Invalid_argument msg ->
                     Diagnostics.error ctx.diags
                       ~code:"E209"
                       ~loc:Diagnostics.no_loc
                       ~message:(Printf.sprintf
                         "%s row %d column %d: expected a number or ISO date (YYYY-MM-DD), got '%s' (%s)"
                         path row_num (n_dims + j + 1) cell msg)
                       ())
                | Some ("instant" | "duration"), None ->
                  Diagnostics.error ctx.diags
                    ~code:"E209"
                    ~loc:Diagnostics.no_loc
                    ~message:(Printf.sprintf
                      "%s row %d column %d: date cell '%s' needs a top-level `origin = date(...)` to resolve (instant/duration table in an unanchored model)"
                      path row_num (n_dims + j + 1) cell)
                    ()
                | _ ->
                  Diagnostics.error ctx.diags
                    ~code:"E209"
                    ~loc:Diagnostics.no_loc
                    ~message:(Printf.sprintf "%s row %d column %d: expected a number, got '%s'"
                      path row_num (n_dims + j + 1) cell)
                    ()))
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
         pipeline rejects (returns Error) on E211 before the IR is
         serialized. *)
      Array.iter (fun arr ->
        for i = 0 to Array.length arr - 1 do
          if Float.is_nan arr.(i) then arr.(i) <- 0.0
        done
      ) arrays
    end;
    Array.to_list arrays
  in
  match read_csv_rows ctx path
          ~ref_desc:"read()" ~ref_hint_example:"read(\"data/contact.tsv\")"
          ~on_header ~on_row ~on_done with
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

(** Map a (possibly expansion-mangled) symbol back to the source loc of its
    base declaration, by the same prefix convention expansion uses (`base`,
    or `base_<stratum>…`). [Diagnostics.no_loc] if no base decl matches (e.g.
    the symbol is an unknown reference, not a declared name). Generalizes the
    per-transition [tr_loc] the autodiff pass uses. *)
let find_decl_loc ctx ~(decls : 'a list) ~(name_of : 'a -> string)
    ~(loc_of : 'a -> Ast.loc) (name : string) : Diagnostics.loc =
  match
    List.find_opt (fun d -> is_expansion_of ~base:(name_of d) name) decls
  with
  | Some d -> diag_loc_of_ast_ctx ctx (loc_of d)
  | None -> Diagnostics.no_loc

(* Per-kind loc lookups over the retained base declarations. The decls keep
   their source locs and are never overwritten to the expanded set, so these
   resolve an IR symbol (possibly stratified) back to its declaration line. *)
let compartment_loc ctx name =
  find_decl_loc ctx ~decls:ctx.comp_decls
    ~name_of:(fun (c : compartment_decl) -> c.cname)
    ~loc_of:(fun c -> c.cloc) name

let transition_loc ctx name =
  find_decl_loc ctx ~decls:ctx.orig_transitions
    ~name_of:(fun (t : transition_decl) -> t.trname)
    ~loc_of:(fun t -> t.trloc) name

let param_loc ctx name =
  find_decl_loc ctx ~decls:ctx.param_decls
    ~name_of:(function PScalar s -> s.pname | PIndexed s -> s.pname)
    ~loc_of:(function PScalar s -> s.ploc | PIndexed s -> s.ploc) name

let obs_loc ctx name =
  find_decl_loc ctx ~decls:ctx.obs_decls
    ~name_of:(fun (o : obs_decl) -> o.oname)
    ~loc_of:(fun o -> o.oloc) name

(* Resolve a contrast name back to its declaration loc — used by the dimcheck
   contrast-dimension diagnostic (which runs on the IR and has no source spans).
   Exact-name match (contrasts are never stratified/expanded). *)
let contrast_loc ctx name =
  match List.find_opt (fun (c : contrast_decl) -> c.cd_name = name) ctx.contrast_decls with
  | Some c -> diag_loc_of_ast_ctx ctx c.cd_loc
  | None -> Diagnostics.no_loc

(* Distinct external data files opened during expansion, as (as-written,
   resolved) pairs in first-seen order. Deduped by resolved path: the same
   file may be read once per stratum level (file-backed indexed forcings,
   DRead dimensions), but the depfile wants the distinct file set. Powers
   `camdlc --emit-deps`. *)
let reads ctx =
  let seen = Hashtbl.create 16 in
  List.filter (fun (_, resolved) ->
    if Hashtbl.mem seen resolved then false
    else (Hashtbl.add seen resolved (); true))
    (List.rev ctx.reads)

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
    | DReactiveInterventions rxs -> ctx.reactive_decls <- List.rev_append rxs ctx.reactive_decls
    | DQuantities qs     -> ctx.quantity_decls <- List.rev_append qs ctx.quantity_decls
    | DContrasts cs      -> ctx.contrast_decls <- List.rev_append cs ctx.contrast_decls
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
  ctx.quantity_decls <- List.rev ctx.quantity_decls;
  ctx.contrast_decls <- List.rev ctx.contrast_decls;
  ctx.orig_transitions <- ctx.transitions

(* ── Staged-residence (`via`) lowering pre-pass ──────────────────────────────

   A `via law(...)` transition desugars to EXACTLY the manual stratified-
   `consecutive` staging a user writes today (the committed golden
   `ocaml/golden/seir_erlang.camdl` is the target form). This pass runs on the
   raw AST in [ctx] AFTER [collect_declarations] and BEFORE [resolve_dimensions]
   / [check_declaration_names], so every downstream phase (stratification,
   `consecutive` expansion, the bare-name `PopSum` rule, the require-full-index
   stoichiometry rule, autodiff) sees ordinary compartments + transitions and
   runs unchanged. `via` adds zero new IR algebra — it is a macro over existing
   AST nodes (staged-residence proposal, 2026-06-26 §5).

   Scope:
   - `erlang` and `hyper_erlang` are both lowered; any other law (`coxian`,
     `fixed`, …) → E243. `hyper_erlang` on an already-stratified source is a
     later sub-phase → E248.
   - The source may itself be stratified (age × stage). A BARE reference to the
     staged source sums over stages for free (PopSum); a PARTIAL-index reference
     (`I[a]` in an age FOI) is rewritten by the pass into the explicit stage-sum
     `sum(__s in __stage, I[a, __s])` (proposal §7, [sum_staged_refs]). *)

(* A `to = <compartment>` branch destination is written as a bare compartment
   identifier — `to = D` or `to = D[a]`. Lift it to a [stoich_ref]. Anything
   else (a number, an arithmetic expr) is rejected: a destination is a
   compartment, not a value. *)
let stoich_ref_of_to_expr (e : expr) : (stoich_ref, string) result =
  match e with
  | EIdent (n, _)   -> Ok (n, [])
  | EIndex (n, idx, _) -> Ok (n, idx)
  | _ -> Error "must be a compartment name (e.g. `to = D` or `to = D[a]`)"

(* Build one validated [hyper_branch] from a `branch(...)` call's keyword args.
   Mirrors the erlang validation (stages = pos-int; exactly one of mean/rate)
   plus the branch-only keywords (`label` required; `weight`/`to` optional). An
   unrecognized keyword is rejected (no loose semantics). Returns [None] (after
   firing a located diagnostic naming the transition) on any failure; the
   cross-branch weight/label/count rules are applied by the caller. *)
let hyper_branch_of_call ctx (tr : transition_decl) (bargs : (string * expr) list)
    : hyper_branch option =
  let err ~code ~message ?hint () =
    Diagnostics.error ctx.diags ~code
      ~loc:(diag_loc_of_ast_ctx ctx tr.trloc) ~message ?hint ();
    None
  in
  (* A branch label, for diagnostics, before we know it is valid. *)
  let label_str =
    match List.assoc_opt "label" bargs with
    | Some (EIdent (s, _)) -> s
    | _ -> "<unlabeled>"
  in
  let known = [ "label"; "weight"; "stages"; "mean"; "rate"; "to" ] in
  let unknown = List.filter (fun (k, _) -> not (List.mem k known)) bargs in
  match unknown with
  | (k, _) :: _ ->
    let what = if k = "" then "a positional argument" else Printf.sprintf "keyword '%s'" k in
    err ~code:"E259"
      ~message:(Printf.sprintf
        "transition '%s': hyper_erlang branch '%s' has %s" tr.trname label_str what)
      ~hint:"a branch takes `label`, `stages`, exactly one of `mean` / `rate`, \
             and optionally `weight` and `to`" ()
  | [] ->
    (* label: a required bare identifier. *)
    let label_res = match List.assoc_opt "label" bargs with
      | Some (EIdent (s, _)) -> Some s
      | Some _ ->
        err ~code:"E259"
          ~message:(Printf.sprintf
            "transition '%s': hyper_erlang branch `label` must be a bare name"
            tr.trname)
          ~hint:"e.g. branch(label = fatal, ...)" ()
      | None ->
        err ~code:"E259"
          ~message:(Printf.sprintf
            "transition '%s': a hyper_erlang branch is missing `label`" tr.trname)
          ~hint:"every branch needs a distinct `label` (it names the per-branch \
                 stage compartments)" ()
    in
    (* stages: a positive-integer literal (same rule as erlang, E244). *)
    let stages_res = match List.assoc_opt "stages" bargs with
      | None ->
        err ~code:"E244"
          ~message:(Printf.sprintf
            "transition '%s': hyper_erlang branch '%s' requires \
             `stages = <positive integer>`" tr.trname label_str)
          ~hint:"e.g. branch(label = fatal, stages = 3, mean = 8 'days)" ()
      | Some (EConst f) ->
        (match Pos_int.of_float f with
         | Ok pi -> Some pi
         | Error why ->
           err ~code:"E244"
             ~message:(Printf.sprintf
               "transition '%s': hyper_erlang branch '%s' `stages` %s"
               tr.trname label_str why)
             ~hint:"`stages` is the number of sub-stages (model structure), a \
                    fixed positive integer — not a fittable parameter" ())
      | Some _ ->
        err ~code:"E244"
          ~message:(Printf.sprintf
            "transition '%s': hyper_erlang branch '%s' `stages` must be a \
             positive-integer literal" tr.trname label_str)
          ~hint:"`stages` sets how many compartments exist; it cannot be a \
                 parameter or an expression" ()
    in
    (* mean XOR rate (same rule as erlang, E245). *)
    let mean_res =
      match List.assoc_opt "mean" bargs, List.assoc_opt "rate" bargs with
      | Some m, None -> Some (Mean m)
      | None, Some r -> Some (Rate r)
      | Some _, Some _ ->
        err ~code:"E245"
          ~message:(Printf.sprintf
            "transition '%s': hyper_erlang branch '%s' sets both `mean` and \
             `rate`; give exactly one" tr.trname label_str)
          ~hint:"`rate` is 1/`mean` — pick the one you have" ()
      | None, None ->
        err ~code:"E245"
          ~message:(Printf.sprintf
            "transition '%s': hyper_erlang branch '%s' sets neither `mean` nor \
             `rate`; give exactly one" tr.trname label_str)
          ~hint:"e.g. branch(label = fatal, stages = 3, mean = 8 'days)" ()
    in
    (* weight: optional (None on the last branch ⇒ implicit). The cross-branch
       last-only rule is checked by the caller. *)
    let weight = List.assoc_opt "weight" bargs in
    (* to: optional per-branch destination, a bare compartment ref. *)
    let to_res = match List.assoc_opt "to" bargs with
      | None -> Some None
      | Some e ->
        (match stoich_ref_of_to_expr e with
         | Ok r -> Some (Some r)
         | Error why ->
           err ~code:"E257"
             ~message:(Printf.sprintf
               "transition '%s': hyper_erlang branch '%s' `to` %s"
               tr.trname label_str why) ())
    in
    (match label_res, stages_res, mean_res, to_res with
     | Some hb_label, Some hb_stages, Some hb_mean, Some hb_to ->
       Some { hb_weight = weight; hb_stages; hb_mean; hb_to; hb_label }
     | _ -> None)

(* Build the typed, validated [via_spec] from the raw [via_call]. Each failure
   is a located diagnostic naming the transition; on the error path we return
   [None] so the caller skips the rewrite (the compile aborts at phase end). *)
let via_spec_of_call ctx (tr : transition_decl) ((law, args) : via_call)
    : via_spec option =
  let err ~code ~message ?hint () =
    Diagnostics.error ctx.diags ~code
      ~loc:(diag_loc_of_ast_ctx ctx tr.trloc) ~message ?hint ();
    None
  in
  match law with
  | "erlang" ->
    (* Reject any keyword that is not `stages` / `mean` / `rate` — a `weight`
       or a misspelling must not be silently dropped (no loose semantics). *)
    let unknown =
      List.filter (fun (k, _) ->
        not (List.mem k [ "stages"; "mean"; "rate" ])) args
    in
    (match unknown with
     | (k, _) :: _ ->
       err ~code:"E247"
         ~message:(Printf.sprintf
           "transition '%s': erlang(...) has no keyword '%s'" tr.trname k)
         ~hint:"erlang takes `stages` and exactly one of `mean` / `rate`" ()
     | [] ->
       (* stages: a positive-integer literal (a pos_int, checked at construction). *)
       let stages_res =
         match List.assoc_opt "stages" args with
         | None ->
           err ~code:"E244"
             ~message:(Printf.sprintf
               "transition '%s': erlang(...) requires `stages = <positive integer>`"
               tr.trname)
             ~hint:"e.g. erlang(stages = 3, mean = 7 'days)" ()
         | Some (EConst f) ->
           (match Pos_int.of_float f with
            | Ok pi -> Some pi
            | Error why ->
              err ~code:"E244"
                ~message:(Printf.sprintf
                  "transition '%s': erlang `stages` %s" tr.trname why)
                ~hint:"`stages` is the number of sub-stages (model structure), \
                       a fixed positive integer — not a fittable parameter" ())
         | Some _ ->
           err ~code:"E244"
             ~message:(Printf.sprintf
               "transition '%s': erlang `stages` must be a positive-integer \
                literal" tr.trname)
             ~hint:"`stages` sets how many compartments exist; it cannot be a \
                    parameter or an expression" ()
       in
       (* mean XOR rate: exactly one present. *)
       let mean_res =
         match List.assoc_opt "mean" args, List.assoc_opt "rate" args with
         | Some m, None -> Some (Mean m)
         | None, Some r -> Some (Rate r)
         | Some _, Some _ ->
           err ~code:"E245"
             ~message:(Printf.sprintf
               "transition '%s': erlang(...) sets both `mean` and `rate`; \
                give exactly one" tr.trname)
             ~hint:"`rate` is 1/`mean` — pick the one you have" ()
         | None, None ->
           err ~code:"E245"
             ~message:(Printf.sprintf
               "transition '%s': erlang(...) sets neither `mean` nor `rate`; \
                give exactly one" tr.trname)
             ~hint:"e.g. erlang(stages = 3, mean = 7 'days) or \
                    erlang(stages = 3, rate = sigma)" ()
       in
       (match stages_res, mean_res with
        | Some stages, Some mean -> Some (Erlang { stages; mean })
        | _ -> None))
  | "hyper_erlang" ->
    (* A finite mixture of Erlang chains, branched at entry. Every argument must
       be a `branch(...)` call: the law itself takes NO bare keywords (no loose
       semantics — `hyper_erlang(stages = 3, ...)` is a mistake, the stages
       belong on a branch). *)
    let non_branch =
      List.filter (fun (k, e) -> match k, e with
        | "", EFuncCall ("branch", _) -> false
        | _ -> true) args
    in
    (match non_branch with
     | (k, _) :: _ ->
       let what = if k = "" then "a non-`branch(...)` argument"
                  else Printf.sprintf "keyword '%s'" k in
       err ~code:"E259"
         ~message:(Printf.sprintf
           "transition '%s': hyper_erlang(...) takes only `branch(...)` arguments \
            but has %s" tr.trname what)
         ~hint:"write hyper_erlang(branch(label = ..., stages = ..., mean = ...), \
                branch(...)); `stages`/`mean`/`weight`/`to` go on each branch, not \
                on hyper_erlang itself" ()
     | [] ->
       (* Parse each branch independently (each accumulating its own diagnostics),
          then apply the cross-branch rules (≥ 2 branches, distinct labels, only
          the LAST branch may omit `weight`). [filter_map] returns only the
          well-formed branches; a branch error still fires (the compile aborts at
          phase end), so we never build a HyperErlang from a partially-valid set:
          if ANY branch failed we return None. *)
       let branch_calls = List.map snd args in
       let parsed = List.map (fun e -> match e with
         | EFuncCall ("branch", bargs) -> hyper_branch_of_call ctx tr bargs
         | _ -> None  (* unreachable: filtered above *)) branch_calls
       in
       if List.exists Option.is_none parsed then None
       else begin
         let branches = List.filter_map (fun x -> x) parsed in
         (* ≥ 2 branches (a 1-branch mixture is an erlang; a 0-branch is empty). *)
         let enough =
           if List.length branches >= 2 then true
           else (ignore (err ~code:"E255"
             ~message:(Printf.sprintf
               "transition '%s': hyper_erlang(...) needs at least 2 branches \
                (a single branch is an ordinary erlang)" tr.trname)
             ~hint:"use `via erlang(stages = ..., mean = ...)` for a single chain"
             ()); false)
         in
         (* Distinct labels (the per-branch stage compartment names derive from
            them: `<src>__<label>__i`). *)
         let labels = List.map (fun b -> b.hb_label) branches in
         let distinct =
           if List.length (List.sort_uniq compare labels) = List.length labels
           then true
           else (ignore (err ~code:"E258"
             ~message:(Printf.sprintf
               "transition '%s': hyper_erlang(...) branches have duplicate labels; \
                each `branch(label = ...)` must be distinct" tr.trname)
             ~hint:"the labels name the per-branch stage compartments, so they \
                    must be unique" ()); false)
         in
         (* Only the LAST branch may omit `weight` (⇒ 1 − Σ others). A non-last
            branch missing `weight`, or the last branch HAVING one, is an error. *)
         let n = List.length branches in
         let weight_ok =
           List.mapi (fun i b ->
             let is_last = i = n - 1 in
             match b.hb_weight, is_last with
             | Some _, false -> true
             | None,   true  -> true
             | None,   false ->
               ignore (err ~code:"E256"
                 ~message:(Printf.sprintf
                   "transition '%s': hyper_erlang branch '%s' has no `weight`; \
                    only the LAST branch may omit it (⇒ 1 − Σ of the others)"
                   tr.trname b.hb_label)
                 ~hint:"give every branch but the last an explicit `weight = ...`"
                 ()); false
             | Some _, true ->
               ignore (err ~code:"E256"
                 ~message:(Printf.sprintf
                   "transition '%s': the LAST hyper_erlang branch '%s' must NOT \
                    set `weight` — it is 1 − Σ of the others, so the mixture is \
                    normalized by construction" tr.trname b.hb_label)
                 ~hint:"drop `weight` from the last branch" ()); false
           ) branches |> List.for_all (fun x -> x)
         in
         (* Literal weights must be probabilities in [0, 1], and if EVERY explicit
            weight is constant-foldable their sum must be ≤ 1 (so the implicit last
            weight 1 − Σ stays ≥ 0). A `: probability` param weight is bounded by the
            param system; an unfoldable expression is left to that layer. An
            out-of-range constant gives a NEGATIVE entry rate / initial population. *)
         let rec weight_const = function
           | EConst c -> Some c
           | EBinOp (Add, a, b) -> (match weight_const a, weight_const b with Some x, Some y -> Some (x +. y) | _ -> None)
           | EBinOp (Sub, a, b) -> (match weight_const a, weight_const b with Some x, Some y -> Some (x -. y) | _ -> None)
           | EBinOp (Mul, a, b) -> (match weight_const a, weight_const b with Some x, Some y -> Some (x *. y) | _ -> None)
           | EBinOp (Div, a, b) -> (match weight_const a, weight_const b with Some x, Some y when y <> 0.0 -> Some (x /. y) | _ -> None)
           | _ -> None
         in
         let explicit  = List.filter_map (fun b -> b.hb_weight) branches in
         let folded    = List.map weight_const explicit in
         let any_out   = List.exists (function Some w -> w < 0.0 || w > 1.0 | None -> false) folded in
         let all_const = explicit <> [] && List.for_all Option.is_some folded in
         let sum_const = List.fold_left (fun acc -> function Some w -> acc +. w | None -> acc) 0.0 folded in
         let sum_over  = all_const && sum_const > 1.0 +. 1e-9 in
         let weights_in_range =
           if any_out || sum_over then
             (ignore (err ~code:"E225"
               ~message:(Printf.sprintf
                 "transition '%s': hyper_erlang(...) branch weights must be \
                  probabilities in [0, 1] summing to <= 1 (the implicit last weight \
                  is 1 - sum of the others)" tr.trname)
               ~hint:"a weight outside [0,1] or weights summing past 1 give a \
                      negative entry rate / negative initial population" ()); false)
           else true
         in
         if enough && distinct && weight_ok && weights_in_range
         then Some (HyperErlang { branches })
         else None
       end)
  | _ ->
    (* Any other law (`coxian`, `approx_gamma`, `fixed`, …) is not yet lowered. *)
    err ~code:"E243"
      ~message:(Printf.sprintf
        "transition '%s': staged-residence `via %s(...)` is not yet supported"
        tr.trname law)
      ~hint:"the laws shipped so far are `erlang(stages, mean | rate)` and \
             `hyper_erlang(branch(...), ...)`; other laws are not implemented \
             yet — express the residence with manual sub-stage compartments for \
             now" ()

(* The single-exit invariant (proposal §3): a staged compartment must be drained
   by EXACTLY ONE transition. A second draining transition (another `via`, or an
   ordinary `@` exit racing with the dwell) is the competing-exit case (§7),
   rejected with a diagnostic naming the compartment. A transition drains a
   compartment when that compartment is a source with no matching destination
   appearance (so a within-compartment self-loop does not count). *)
let drains_compartment (tr : transition_decl) (comp : string) : bool =
  let is_src = List.exists (fun (c, _) -> c = comp) tr.trsrc in
  let is_dst = match tr.trdst with
    | DstSum refs        -> List.exists (fun (c, _) -> c = comp) refs
    | DstBranch branches -> List.exists (fun ((c, _), _) -> c = comp) branches
  in
  is_src && not is_dst

(* Rewrite the destination form so any inflow to [from_comp] lands in its first
   stage. Once the source is staged, the require-full-index rule rejects a
   destination that stops one dimension short of the stage axis, so the stage
   level must be appended. [n_pre] is the number of index positions [from_comp]
   carried BEFORE staging (0 for an unstratified compartment, 1 for an age-
   stratified one, …); the stage dimension is appended last, so an inflow ref
   that already supplies all [n_pre] pre-staging indices gets `stage1` appended:
   `S --> I` ⇒ `I[s1]`, `S[a] --> I[a]` ⇒ `I[a, s1]`. A reference with fewer than
   [n_pre] indices is left alone — it is the under-indexed case the existing
   require-full-index diagnostic already rejects. *)
let redirect_dest_to_stage1 (dst : destination_form) (from_comp : string)
    (n_pre : int) (stage1 : string) : destination_form =
  let redirect ((c, items) : stoich_ref) : stoich_ref =
    if c = from_comp && List.length items = n_pre then
      (c, items @ [ IPosn (EIdent (stage1, dummy_loc)) ])
    else (c, items)
  in
  match dst with
  | DstSum refs        -> DstSum (List.map redirect refs)
  | DstBranch branches -> DstBranch (List.map (fun (r, w) -> (redirect r, w)) branches)

(* Rewrite every RATE-POSITION reference to a now-staged compartment [src] so it
   sums over the freshly-added stage dimension. After staging, `src` gains a
   trailing stage axis (`I` → `I[…, stage]`); a reference that supplied all
   [n_pre] of its pre-staging indices but no stage index is partial and would
   resolve to a non-existent cell (`I_b` when the cells are `I_b_s1 …`). The pass
   created the stages, so it owns this rewrite (proposal §7, the declined general
   partial-index case): `I[b]` ⇒ `sum(__s in __stage, I[b, __s])`. A BARE `src`
   (`EIdent`, no indices) is left alone — the existing bare-name rule already
   turns it into a `PopSum` over all cells, stages included. An under-indexed
   reference (fewer than [n_pre] indices) is also left alone: that is the general
   partial-index notation, still declined, and keeps its existing diagnostic.

   [sum_var] is a fresh bound variable (collision-free, derived from the reserved
   [dim_name]) so the synthesized `sum` cannot capture or be captured by a user
   index. The walk recurses through every expr node, including nested `sum`
   bodies (the per-age FOI `sum(b in age, … I[b] …)`) and `let` bodies. *)
let rec sum_staged_refs ~src ~n_pre ~dim_name ~sum_var (e : expr) : expr =
  let recur = sum_staged_refs ~src ~n_pre ~dim_name ~sum_var in
  let recur_item = function
    | IPosn e        -> IPosn (recur e)
    | INamed (n, e)  -> INamed (n, recur e)
  in
  match e with
  | EConst _ | EUnit _ | EIdent _ | EObsAccess _ | ERunMember _ -> e
  | EIndex (n, items, l) when n = src && List.length items = n_pre ->
    (* The partial reference to the staged compartment: append the stage index
       and wrap in a sum over the stage dimension. Recurse into the existing
       index exprs first (they may themselves reference the staged compartment,
       though in practice they are bare loop vars). *)
    let items' = List.map recur_item items in
    let staged =
      EIndex (n, items' @ [ IPosn (EIdent (sum_var, dummy_loc)) ], l)
    in
    ESum (sum_var, dim_name, None, staged)
  | EIndex (n, items, l) -> EIndex (n, List.map recur_item items, l)
  | EBinOp (op, l, r) -> EBinOp (op, recur l, recur r)
  | EUnOp (op, e)     -> EUnOp (op, recur e)
  | ESum (v, d, g, b) -> ESum (v, d, g, recur b)
  | ECond (p, t, f)   -> ECond (recur p, recur t, recur f)
  | EFuncCall (f, args) -> EFuncCall (f, List.map (fun (k, e) -> (k, recur e)) args)
  | EList es          -> EList (List.map recur es)
  | ERange (lo, hi)   -> ERange (recur lo, recur hi)

(* Rewrite every RATE-POSITION reference to a [src] staged by `hyper_erlang` into
   an explicit Add-chain over ALL its per-branch flat stage compartments. Unlike
   erlang (one stage dimension → the bare-name `PopSum` rule sums for free),
   hyper_erlang generates FLAT per-branch compartments (`I__fatal__1`,
   `I__recover__1`, …) that are NOT one dimension, so a bare `I` no longer
   resolves to any compartment and must be summed by name here. [cells] is the
   ordered list of every per-branch stage compartment created for [src]; the
   rewrite preserves their order so the Add-chain folds left-to-right (matching
   the OCaml Add-chain order). Only a BARE `EIdent src` is rewritten — the chain
   transitions reference the flat cells directly (a different name), and a
   `src`-indexed reference cannot occur (stratified hyper_erlang is deferred). *)
let rec sum_hyper_refs ~src ~(cells : string list) (e : expr) : expr =
  let recur = sum_hyper_refs ~src ~cells in
  let recur_item = function
    | IPosn e       -> IPosn (recur e)
    | INamed (n, e) -> INamed (n, recur e)
  in
  match e with
  | EIdent (n, _) when n = src ->
    (* The Add-chain `cell_1 + cell_2 + … + cell_m`, left-folded. [cells] is
       non-empty (every branch has ≥ 1 stage), so [List.tl]/[List.hd] are safe. *)
    let refs = List.map (fun c -> EIdent (c, dummy_loc)) cells in
    List.fold_left (fun acc r -> EBinOp (Add, acc, r)) (List.hd refs) (List.tl refs)
  | EConst _ | EUnit _ | EIdent _ | EObsAccess _ | ERunMember _ -> e
  | EIndex (n, items, l) -> EIndex (n, List.map recur_item items, l)
  | EBinOp (op, l, r) -> EBinOp (op, recur l, recur r)
  | EUnOp (op, e)     -> EUnOp (op, recur e)
  | ESum (v, d, g, b) -> ESum (v, d, g, recur b)
  | ECond (p, t, f)   -> ECond (recur p, recur t, recur f)
  | EFuncCall (f, args) -> EFuncCall (f, List.map (fun (k, e) -> (k, recur e)) args)
  | EList es          -> EList (List.map recur es)
  | ERange (lo, hi)   -> ERange (recur lo, recur hi)

(* Lower every `via erlang(...)` transition in [ctx] into the manual staged form.
   Mutates ctx.dim_decls / ctx.stratifies / ctx.transitions / ctx.init_entries,
   and — when the staged source is itself stratified — ctx.let_bindings /
   ctx.obs_decls / ctx.balance_decl, to rewrite partial references to the staged
   compartment into stage-sums ([sum_staged_refs]). The original `via` transition
   is replaced by the consecutive chain + exit, so no `via` transition survives to
   reach [expand_transitions_counted] (its E243 placeholder becomes unreachable). *)
let lower_via_transitions ctx =
  (* Collect the `via` transitions and their typed specs first; bail out early
     if there are none (the common, golden-neutral path: zero AST touched). A
     spec that fails validation carries [None]: the transition is still REMOVED
     (so it never reaches E243), but no chain is synthesized — a diagnostic has
     already fired and the compile aborts at phase end. *)
  let via_trs =
    List.filter_map (fun tr ->
      match tr.trdyn with
      | Rate _   -> None
      | Via call -> Some (tr, via_spec_of_call ctx tr call)
    ) ctx.transitions
  in
  if via_trs = [] then ()
  else begin
    (* Validate each `via` transition's STRUCTURE: a single, real source
       (stratified or not), drained by exactly this one transition (single-exit,
       §3). A stratified source is staged by COMPOSING the stage dimension onto
       its existing stratification (age × stage); rate-position references to it
       are rewritten to sum over the stages by the pass itself (proposal §7).
       Returns whether the transition is structurally well-formed; every problem
       surfaces (a diagnostic fires) even on a transition we then skip. We only
       SYNTHESIZE the chain for transitions that pass BOTH spec and structure —
       so a rejected `via` produces just its own diagnostic, never a cascade of
       follow-on E100/E272/E277 noise from a half-applied stratification. *)
    let structurally_ok (tr : transition_decl) : bool =
      let loc = diag_loc_of_ast_ctx ctx tr.trloc in
      match tr.trsrc with
      | [ (src, _) ] ->
        begin
          (* Single-exit: no OTHER transition may drain this source. *)
          let other_drainers =
            List.filter (fun (other : transition_decl) ->
              other.trname <> tr.trname && drains_compartment other src)
              ctx.transitions
          in
          match other_drainers with
          | other :: _ ->
            Diagnostics.error ctx.diags ~code:"E246" ~loc
              ~message:(Printf.sprintf
                "compartment '%s' has a staged residence (`via` on '%s') but \
                 is also drained by transition '%s'" src tr.trname other.trname)
              ~hint:"a staged compartment must be drained by exactly one `via` \
                     transition; a second exit racing with the dwell is the \
                     competing-exit case — express it with manual per-stage \
                     compartments for now"
              ();
            false
          | [] -> true
        end
      | _ ->
        Diagnostics.error ctx.diags ~code:"E249" ~loc
          ~message:(Printf.sprintf
            "transition '%s': `via` requires a single source compartment to \
             stage" tr.trname)
          ~hint:"a staged residence stages one compartment; write one source"
          ();
        false
    in
    (* The transitions we actually lower: both spec-valid and structure-valid.
       Compute the structural check for every via transition (so all errors
       surface) but keep only the fully-valid ones for synthesis. *)
    let to_lower =
      List.filter_map (fun (tr, spec) ->
        let struct_ok = structurally_ok tr in
        match spec with
        | Some s when struct_ok -> Some (tr, s)
        | _ -> None
      ) via_trs
    in

    (* For each well-formed `via erlang`, synthesize the stage dimension, the
       stratify entry, the consecutive chain + exit, and redirect inflow/init. We
       build the SAME AST a user writes manually, so all downstream machinery is
       identical. Only the fully-valid transitions ([to_lower]) are synthesized;
       a rejected `via` is removed below without a chain. *)
    List.iter (fun (tr, spec) ->
      match tr.trsrc, spec with
      | [ (src, src_pre_items) ], Erlang { stages; mean } ->
        let k = Pos_int.to_int stages in
        if k = 1 then begin
          (* stages = 1 is the ordinary exponential dwell (proposal §4): Erlang(1)
             IS the exponential — not a no-op, but no sub-staging either. Replace
             the `via` transition in place with the plain exponential exit
             `src --> dst @ coeff·src`, coeff = `rate` (or `1/mean`). The source
             stays unstaged (no stage dimension, no inflow/init redirect), so the
             k=1 member of a `stages = 1,2,3,…` sweep is byte-identical to writing
             `@ rate` directly. *)
          let coeff = match mean with
            | Rate r -> r
            | Mean t -> EBinOp (Div, EConst 1.0, t)
          in
          let src_ref = match src_pre_items with
            | []    -> EIdent (src, dummy_loc)
            | items -> EIndex (src, items, dummy_loc)
          in
          let exit_tr = { tr with trdyn = Rate (EBinOp (Mul, coeff, src_ref)) } in
          ctx.transitions <- List.map (fun (t : transition_decl) ->
            if t.trname = tr.trname then exit_tr else t) ctx.transitions
        end else begin
        (* The source's index positions BEFORE staging: 0 for an unstratified
           compartment, 1 for an age-stratified one, … Counted here, before the
           stage dimension is appended to [ctx.stratifies], so it counts only the
           PRE-existing strata (the same `sonly` filter [comp_dims] applies). The
           stage axis is appended LAST, so every staged reference to [src] carries
           `n_pre` inherited indices + the stage. *)
        let n_pre =
          List.length (List.filter (fun (sd : stratify_decl) ->
            match sd.sonly with None -> true | Some only -> List.mem src only)
            ctx.stratifies)
        in
        (* [src_pre_items] (bound by the match) is the via transition's own
           source indices (e.g. `[a]` from `recovery[a in age] : I[a] --> R[a]`).
           Inherited by the synthesized chain and exit so each per-stratum
           residence stays within its stratum; the stage index is appended last. *)
        (* Stage levels: a reserved dimension `__<trname>_stage = [s1..sk]`. The
           `__` prefix keeps it out of the user namespace (collision-free). *)
        let dim_name = Printf.sprintf "__%s_stage" tr.trname in
        let stage_levels = List.init k (fun i -> Printf.sprintf "s%d" (i + 1)) in
        let stage1 = List.hd stage_levels in
        let stage_last = List.nth stage_levels (k - 1) in
        (* Fresh stage bound-variable names, derived from the reserved (collision-
           checked) [dim_name] so they cannot capture or be captured by a user
           index var — even one the via transition itself carries (`a`). Three
           distinct vars: the chain's `[(s, s_next) in consecutive(...)]` pair and
           the rewrite's `sum(s in ...)` variable. *)
        let chain_var      = dim_name ^ "_i" in
        let chain_var_next = dim_name ^ "_n" in
        let sum_var        = dim_name ^ "_s" in
        (* Per-stage rate COEFFICIENT: `Rate ρ ⇒ k·ρ`, `Mean τ ⇒ k/τ`. *)
        let k_const = EConst (float_of_int k) in
        let coeff = match mean with
          | Rate r -> EBinOp (Mul, k_const, r)
          | Mean t -> EBinOp (Div, k_const, t)
        in
        (* Per-stage propensity `coeff * src[<inherited indices>, stage]`, where
           the stage index is a bound var (the chain) or a concrete level (the
           exit). For an unstratified source this is exactly the hand-written
           `3 * sigma * E[s]`; for an age-stratified one, `3 * gamma * I[a, s]`. *)
        let stage_pop stage_item = EIndex (src, src_pre_items @ [ stage_item ], dummy_loc) in
        let stage_rate stage_item = EBinOp (Mul, coeff, stage_pop stage_item) in

        (* 1. Register the stage dimension (picked up by resolve_dimensions). *)
        ctx.dim_decls <- ctx.dim_decls @ [
          { dename = dim_name; desrc = DInline stage_levels; dedoc = None } ];
        (* 2. Stratify the source compartment by the stage dimension. This
              COMPOSES with any existing stratification of [src] (age × stage),
              since [comp_dims] reads every applicable stratify decl in order. *)
        ctx.stratifies <- ctx.stratifies @ [ { sdim = dim_name; sonly = Some [ src ] } ];

        (* 3. The consecutive chain and the exit. Both INHERIT the via
              transition's existing index bindings and source-stoich indices and
              append the stage; the chain advances the stage within the stratum
              (`I[a, s] --> I[a, s_next]`), the exit drains the last stage to the
              original destination (`I[a, s_last] --> R[a]`). For an unstratified
              source these reduce to the Phase-2 chain `E[s] --> E[s_next]`. *)
        let chain_tr = {
          trname    = Printf.sprintf "%s_stage" tr.trname;
          trindices = tr.trindices @ [ IConsec (chain_var, chain_var_next, dim_name) ];
          trsrc     = [ (src, src_pre_items @ [ IPosn (EIdent (chain_var, dummy_loc)) ]) ];
          trdst     = DstSum [ (src, src_pre_items @ [ IPosn (EIdent (chain_var_next, dummy_loc)) ]) ];
          trdyn     = Rate (stage_rate (IPosn (EIdent (chain_var, dummy_loc))));
          trguard   = tr.trguard;
          trlineage = false;
          trdoc     = None;
          trloc     = tr.trloc;
        } in
        let last_item = IPosn (EIdent (stage_last, dummy_loc)) in
        let exit_tr = {
          tr with
          trsrc = [ (src, src_pre_items @ [ last_item ]) ];
          trdyn = Rate (stage_rate last_item);
          (* trdst, trindices unchanged: the exit keeps the original destination
             (`R[a]`) and index bindings (`[a in age]`). *)
        } in
        (* 4. Replace the original `via` transition with [chain; exit], and
              redirect every OTHER transition whose destination is the staged
              source to land in stage 1 (`S[a] --> I[a]` ⇒ `I[a, s1]`). *)
        ctx.transitions <- List.concat_map (fun (t : transition_decl) ->
          if t.trname = tr.trname then [ chain_tr; exit_tr ]
          else [ { t with trdst = redirect_dest_to_stage1 t.trdst src n_pre stage1 } ]
        ) ctx.transitions;
        (* 5. Redirect `init { E = … }` / `init { I[a] = … }` (the full pre-staging
              index, no stage) to land in stage 1. *)
        ctx.init_entries <- List.map (fun (ie : init_entry) ->
          if ie.icomp = src && ie.ibindings = []
             && List.length ie.iindices = n_pre then
            { ie with iindices = ie.iindices @ [ IPosn (EIdent (stage1, dummy_loc)) ] }
          else ie
        ) ctx.init_entries;
        (* 6. The partial-reference rewrite (the crux for a stratified source).
              Every RATE-POSITION reference to [src] that supplies its `n_pre`
              pre-staging indices but no stage index — `I[b]` in the per-age FOI,
              in `let N_local`, in observations, in `balance` — is rewritten to
              sum over the stages: `I[b]` ⇒ `sum(__s in __stage, I[b, __s])`. A
              BARE `src` is left alone (the bare-name rule sums all cells via
              PopSum). The synthesized chain/exit above already carry the full
              `n_pre + 1` index, so they are immune to this rewrite (the predicate
              fires only at exactly `n_pre` indices). Runs over every expr-bearing
              declaration so no reference to the staged compartment escapes. *)
        let rw e = sum_staged_refs ~src ~n_pre ~dim_name ~sum_var e in
        let rw_dst = function
          | DstSum refs ->
            DstSum (List.map (fun (c, items) ->
              (c, List.map (function IPosn e -> IPosn (rw e)
                                   | INamed (n, e) -> INamed (n, rw e)) items)) refs)
          | DstBranch branches ->
            DstBranch (List.map (fun (r, w) ->
              let (c, items) = r in
              ((c, List.map (function IPosn e -> IPosn (rw e)
                                    | INamed (n, e) -> INamed (n, rw e)) items), rw w))
              branches)
        in
        let rw_dyn = function
          | Rate e -> Rate (rw e)
          | Via _ as v -> v   (* unlowered via transitions keep their raw call *)
        in
        ctx.transitions <- List.map (fun (t : transition_decl) ->
          { t with
            trsrc = List.map (fun (c, items) ->
              (c, List.map (function IPosn e -> IPosn (rw e)
                                   | INamed (n, e) -> INamed (n, rw e)) items)) t.trsrc;
            trdst = rw_dst t.trdst;
            trdyn = rw_dyn t.trdyn }
        ) ctx.transitions;
        ctx.let_bindings <- List.map (fun (lb : let_binding) ->
          { lb with lbody = rw lb.lbody }) ctx.let_bindings;
        ctx.init_entries <- List.map (fun (ie : init_entry) ->
          { ie with ivalue = rw ie.ivalue }) ctx.init_entries;
        ctx.balance_decl <- Option.map (fun (bd : balance_decl) ->
          { bd with bexpr = rw bd.bexpr }) ctx.balance_decl;
        ctx.obs_decls <- List.map (fun (od : obs_decl) ->
          let oprojection = match od.oprojection with
            | Some (ProjDerived e) -> Some (ProjDerived (rw e))
            | other -> other
          in
          let omeasurement = Option.map (fun (om : obs_measurement) ->
            let rw_kwargs = List.map (fun (key, e) -> (key, rw e)) in
            let om_lik = match om.om_lik with
              | LikNegBinomial a  -> LikNegBinomial  (rw_kwargs a)
              | LikPoisson a      -> LikPoisson      (rw_kwargs a)
              | LikNormal a       -> LikNormal       (rw_kwargs a)
              | LikBinomial a     -> LikBinomial     (rw_kwargs a)
              | LikBetaBinomial a -> LikBetaBinomial (rw_kwargs a)
              | LikBernoulli a    -> LikBernoulli    (rw_kwargs a)
            in
            { om with om_lik }) od.omeasurement
          in
          { od with oprojection; omeasurement }
        ) ctx.obs_decls
        end

      | [ (src, src_pre_items) ], HyperErlang { branches } ->
        (* ── hyper_erlang: a finite mixture of Erlang chains, branched at entry ──
           Unlike erlang, the branches have different lengths, so this is NOT one
           stage dimension. We generate FLAT per-branch compartments
           `<src>__<label>__i` and parallel chains, branch the entry into the
           per-branch first stages (weighted), and sum every bare-`src` reference
           over all branch stages by name. Stratified hyper_erlang (an age/space-
           stratified `src`) is DEFERRED — guarded below with E248. *)
        let loc = diag_loc_of_ast_ctx ctx tr.trloc in
        (* DEFER stratified hyper_erlang. [src] is stratified iff some stratify
           decl applies to it; its via transition would also carry `src_pre_items`
           (the per-stratum index). Either is the deferred case. *)
        let src_is_stratified =
          src_pre_items <> []
          || List.exists (fun (sd : stratify_decl) ->
               match sd.sonly with None -> true | Some only -> List.mem src only)
               ctx.stratifies
        in
        if src_is_stratified then
          Diagnostics.error ctx.diags ~code:"E248" ~loc
            ~message:(Printf.sprintf
              "transition '%s': `via hyper_erlang(...)` on the stratified \
               compartment '%s' is not yet supported" tr.trname src)
            ~hint:"hyper_erlang on a stratified compartment is a later sub-phase; \
                   for now use it on an unstratified compartment, or express the \
                   mixture with manual per-stage compartments"
            ()
        else begin
          (* The per-branch weight EXPRESSIONS. Every branch but the last carries
             an explicit `weight`; the last is `1 − Σ others`, so the mixture is
             normalized by construction (validation already enforced this). *)
          let explicit_weights =
            List.filter_map (fun b -> b.hb_weight) branches in
          let weight_of_branch (b : hyper_branch) : expr =
            match b.hb_weight with
            | Some w -> w
            | None ->
              (* last branch: 1 − (w_1 + … + w_{n-1}). *)
              let sum_others = match explicit_weights with
                | []      -> EConst 0.0   (* unreachable: ≥ 2 branches, ≥ 1 explicit *)
                | w :: ws -> List.fold_left (fun a x -> EBinOp (Add, a, x)) w ws
              in
              EBinOp (Sub, EConst 1.0, sum_others)
          in
          (* The transition's arrow target, if any (the same-endpoint default).
             None when the no-arrow form was used (`I via hyper_erlang(...)`) —
             then every branch MUST carry its own `to`. *)
          let arrow_target : stoich_ref option =
            match tr.trdst with
            | DstSum [ r ] -> Some r
            | DstSum []    -> None
            | DstSum _ | DstBranch _ -> None  (* multi/branch arrow rejected below per-branch *)
          in
          (* Resolve a branch's destination: its own `to`, else the arrow target.
             Neither → E257 (a compile error; we substitute [src] as a placeholder
             so synthesis continues and the user sees every error at once). *)
          let dest_of_branch (b : hyper_branch) : stoich_ref =
            match b.hb_to, arrow_target with
            | Some r, _      -> r
            | None,   Some r -> r
            | None,   None   ->
              Diagnostics.error ctx.diags ~code:"E257" ~loc
                ~message:(Printf.sprintf
                  "transition '%s': hyper_erlang branch '%s' has no destination — \
                   it sets no `to` and the transition has no `--> TO` arrow target"
                  tr.trname b.hb_label)
                ~hint:"give the branch a `to = <compartment>`, or put a shared \
                       `--> TO` on the transition arrow"
                ();
              (src, [])  (* placeholder; the error aborts the compile *)
          in
          (* Per-branch flat stage compartment names: `<src>__<label>__i`. The
             `__` (double underscore) keeps them clear of single-`_` stratification
             cells and is collision-checked (these are real comp_decls, so
             [check_declaration_names] rejects any clash with a user name). *)
          let branch_cells (b : hyper_branch) : string list =
            let k = Pos_int.to_int b.hb_stages in
            List.init k (fun i -> Printf.sprintf "%s__%s__%d" src b.hb_label (i + 1))
          in
          (* All per-branch stage cells, in branch order then stage order. The
             bare-`src` reference sums over exactly these (the Add-chain). *)
          let all_cells = List.concat_map branch_cells branches in

          (* 1. Register the flat per-branch stage compartments as new comp_decls
                (the original [src] decl is removed in step 7). They inherit the
                source compartment's kind (Integer/Real). *)
          let src_kind =
            match List.find_opt (fun (cd : compartment_decl) -> cd.cname = src)
                    ctx.comp_decls with
            | Some cd -> cd.ckind | None -> Integer
          in
          let cell_decls =
            List.map (fun name ->
              { cname = name; ckind = src_kind; cdoc = None; cloc = tr.trloc })
              all_cells
          in
          ctx.comp_decls <- ctx.comp_decls @ cell_decls;

          (* 2. Per branch: the chain transitions + the exit. The per-stage rate
                COEFFICIENT is `Rate ρ ⇒ k·ρ`, `Mean τ ⇒ k/τ` (same shape as
                erlang, per branch). The chain advances stage i → i+1; the exit
                drains the last stage to the branch's resolved destination. *)
          let branch_transitions (b : hyper_branch) : transition_decl list =
            let k = Pos_int.to_int b.hb_stages in
            let k_const = EConst (float_of_int k) in
            let coeff = match b.hb_mean with
              | Rate r -> EBinOp (Mul, k_const, r)
              | Mean t -> EBinOp (Div, k_const, t) in
            let cells = branch_cells b in
            let cell i = List.nth cells i in
            let stage_rate name = EBinOp (Mul, coeff, EIdent (name, dummy_loc)) in
            (* The intra-chain steps `cell_i --> cell_{i+1}`. *)
            let chain =
              List.init (k - 1) (fun i ->
                { trname    = Printf.sprintf "%s_%s_stage%d" tr.trname b.hb_label (i + 1);
                  trindices = tr.trindices;
                  trsrc     = [ (cell i, []) ];
                  trdst     = DstSum [ (cell (i + 1), []) ];
                  trdyn     = Rate (stage_rate (cell i));
                  trguard   = tr.trguard;
                  trlineage = false;
                  trdoc     = None;
                  trloc     = tr.trloc })
            in
            (* The exit `cell_{k-1} --> dest`. *)
            let dest = dest_of_branch b in
            let exit_tr =
              { trname    = Printf.sprintf "%s_%s_exit" tr.trname b.hb_label;
                trindices = tr.trindices;
                trsrc     = [ (cell (k - 1), []) ];
                trdst     = DstSum [ dest ];
                trdyn     = Rate (stage_rate (cell (k - 1)));
                trguard   = tr.trguard;
                trlineage = false;
                trdoc     = None;
                trloc     = tr.trloc }
            in
            chain @ [ exit_tr ]
          in
          let synthesized = List.concat_map branch_transitions branches in

          (* 3. The weighted entry DstBranch. Every OTHER transition whose
                destination is [src] is rewritten to branch into the per-branch
                FIRST stages, weighted: `onset : E --> I @ r` becomes
                `onset : E --> { I__b1__1 : w_1, I__b2__1 : w_2, … } @ r`. We emit
                the AST `DstBranch` and let the EXISTING DstBranch lowering scale
                each branch rate by its weight (expander §expand_transitions). *)
          let entry_branch : (stoich_ref * expr) list =
            List.map (fun b ->
              let first_cell = List.hd (branch_cells b) in
              ((first_cell, []), weight_of_branch b)) branches
          in
          let redirect_entry (t : transition_decl) : destination_form =
            (* Replace a `DstSum` mention of [src] (the inflow target) with the
               weighted branch. A transition that does not target [src] is left
               alone. A `DstBranch` inflow into [src] — the branching analogue of
               the multi-destination case — is likewise rejected with a targeted
               E224 (below), not passed through to a confusing downstream E503. *)
            match t.trdst with
            | DstSum refs when List.exists (fun (c, _) -> c = src) refs ->
              let others = List.filter (fun (c, _) -> c <> src) refs in
              if others = [] then DstBranch entry_branch
              else begin
                (* A multi-destination inflow `--> src + X` cannot be split across
                   the mixture's entry branches: weighting src's stages AND keeping
                   X would double src's drain (X as a weight-1 sibling) and decouple
                   X from the branch event. Reject it — the single-destination
                   inflow is the supported pattern. (erlang staging keeps
                   `--> src + X` as one transition, so this is hyper-only.) *)
                Diagnostics.error ctx.diags ~code:"E224"
                  ~loc:(diag_loc_of_ast_ctx ctx t.trloc)
                  ~message:(Printf.sprintf
                    "transition '%s': an inflow into the hyper-staged compartment \
                     '%s' also produces other compartments; a multi-destination \
                     inflow cannot be split across the mixture's entry branches"
                    t.trname src)
                  ~hint:"enter a hyper_erlang-staged source with single-destination \
                         inflows (`--> src`); move the co-products to a separate \
                         transition, or use manual per-stage compartments"
                  ();
                (* Siblings dropped so the rejected lowering stays structurally
                   clean — the E224 error blocks the compile, so it is unused. *)
                DstBranch entry_branch
              end
            | DstBranch brs when List.exists (fun ((c, _), _) -> c = src) brs ->
              (* A branching inflow `--> { src : w, … }` into the hyper-staged
                 source cannot be composed with the mixture's own entry branching.
                 Reject it with the same E224. Redirect the dangling [src] branch
                 to the first stage cell so the (rejected) lowering does not ALSO
                 trip an E503 on the staged-away compartment — the E224 blocks the
                 compile, so this rewrite is never used. *)
              Diagnostics.error ctx.diags ~code:"E224"
                ~loc:(diag_loc_of_ast_ctx ctx t.trloc)
                ~message:(Printf.sprintf
                  "transition '%s': a branching inflow (`--> { … }`) into the \
                   hyper-staged compartment '%s' is not supported; a staged source \
                   must be entered by single-destination inflows (`--> %s`)"
                  t.trname src src)
                ~hint:"move the branch into a separate transition, or use manual \
                       per-stage compartments"
                ();
              let s1 = List.hd (branch_cells (List.hd branches)) in
              DstBranch (List.map (fun ((c, items), w) ->
                if c = src then ((s1, items), w) else ((c, items), w)) brs)
            | _ -> t.trdst
          in

          (* 4. Replace the original `via` transition with the synthesized chains,
                and redirect every OTHER transition's inflow into the staged source
                to the weighted branch. *)
          ctx.transitions <- List.concat_map (fun (t : transition_decl) ->
            if t.trname = tr.trname then synthesized
            else [ { t with trdst = redirect_entry t } ]
          ) ctx.transitions;

          (* 5. Split `init { src = n }` across the branch first-stages by weight:
                `init { src = n }` ⇒ `init { I__b1__1 = n*w_1, I__b2__1 = n*w_2, … }`.
                A plain (unindexed, unbound) init on [src] is the only form a
                hyper-staged unstratified source can carry. *)
          ctx.init_entries <- List.concat_map (fun (ie : init_entry) ->
            if ie.icomp = src && ie.ibindings = [] && ie.iindices = [] then
              List.map (fun b ->
                { ie with
                  icomp   = List.hd (branch_cells b);
                  ivalue  = EBinOp (Mul, ie.ivalue, weight_of_branch b) })
                branches
            else [ ie ]
          ) ctx.init_entries;

          (* 6. Rewrite every bare-`src` rate-position reference into the Add-chain
                over all branch stage cells (the FOI, lets, balance, observations,
                and the new init exprs). The synthesized chains reference the flat
                cells by name, so they are untouched. *)
          let rw e = sum_hyper_refs ~src ~cells:all_cells e in
          let rw_items items =
            List.map (function IPosn e -> IPosn (rw e)
                             | INamed (n, e) -> INamed (n, rw e)) items in
          let rw_dst = function
            | DstSum refs -> DstSum (List.map (fun (c, items) -> (c, rw_items items)) refs)
            | DstBranch brs ->
              DstBranch (List.map (fun ((c, items), w) -> ((c, rw_items items), rw w)) brs)
          in
          let rw_dyn = function
            | Rate e -> Rate (rw e)
            | Via _ as v -> v in
          ctx.transitions <- List.map (fun (t : transition_decl) ->
            { t with
              trsrc = List.map (fun (c, items) -> (c, rw_items items)) t.trsrc;
              trdst = rw_dst t.trdst;
              trdyn = rw_dyn t.trdyn }
          ) ctx.transitions;
          ctx.let_bindings <- List.map (fun (lb : let_binding) ->
            { lb with lbody = rw lb.lbody }) ctx.let_bindings;
          ctx.init_entries <- List.map (fun (ie : init_entry) ->
            { ie with ivalue = rw ie.ivalue }) ctx.init_entries;
          ctx.balance_decl <- Option.map (fun (bd : balance_decl) ->
            { bd with bexpr = rw bd.bexpr }) ctx.balance_decl;
          ctx.obs_decls <- List.map (fun (od : obs_decl) ->
            let oprojection = match od.oprojection with
              | Some (ProjDerived e) -> Some (ProjDerived (rw e))
              | other -> other in
            let omeasurement = Option.map (fun (om : obs_measurement) ->
              let rw_kwargs = List.map (fun (key, e) -> (key, rw e)) in
              let om_lik = match om.om_lik with
                | LikNegBinomial a  -> LikNegBinomial  (rw_kwargs a)
                | LikPoisson a      -> LikPoisson      (rw_kwargs a)
                | LikNormal a       -> LikNormal       (rw_kwargs a)
                | LikBinomial a     -> LikBinomial     (rw_kwargs a)
                | LikBetaBinomial a -> LikBetaBinomial (rw_kwargs a)
                | LikBernoulli a    -> LikBernoulli    (rw_kwargs a)
              in
              { om with om_lik }) od.omeasurement in
            { od with oprojection; omeasurement }
          ) ctx.obs_decls;

          (* 7. Remove the now-replaced base compartment [src] — its flat per-branch
                cells carry the whole population, and a surviving bare `src`
                compartment would shadow the Add-chain (and dangle, since no
                transition fills it). *)
          ctx.comp_decls <- List.filter (fun (cd : compartment_decl) ->
            cd.cname <> src) ctx.comp_decls
        end

      | _ -> ()  (* unreachable: to_lower only holds single-source via specs *)
    ) to_lower;
    (* Drop any `via` transition that survived (a failed spec, or a structurally
       rejected source) so none reaches [expand_transitions_counted]'s E243
       placeholder — its diagnostic already fired and the compile aborts at
       phase end. *)
    ctx.transitions <- List.filter (fun (t : transition_decl) ->
      match t.trdyn with Via _ -> false | Rate _ -> true) ctx.transitions
  end

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
  match read_csv_rows ctx path
          ~ref_desc:"read()" ~ref_hint_example:"read(\"data/levels.tsv\")"
          ~on_header ~on_row ~on_done with
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

(** Expand an indexed-declaration's `<base>_<level>...` names over the
    cartesian product of its declared dims' levels, in row-major order —
    the same name-mangling `resolve_ident_name` / `build_lookup_tables`
    produce for indexed parameters and forcings. A dim with no registered
    levels contributes nothing (the dim error is reported elsewhere). *)
let expand_indexed_decl_names ctx base dims =
  if dims = [] then [base]
  else
    let level_lists = List.map (fun d ->
      match List.assoc_opt d ctx.dim_registry with Some vs -> vs | None -> []) dims in
    if List.exists (fun l -> l = []) level_lists then []
    else
      let rec cart = function
        | [] -> [[]]
        | vs :: rest ->
          let tails = cart rest in
          List.concat_map (fun v -> List.map (fun t -> v :: t) tails) vs
      in
      List.map (fun combo -> String.concat "_" (base :: combo)) (cart level_lists)

(** gh#117: declaration-name validation.

    `build_lookup_tables` populates every namespace table with
    [Hashtbl.replace] — silent last-wins. A duplicate within a namespace
    (`let beta = ...` twice) or the same name across namespaces (a
    `parameter N` and a `let N`) would resolve to whichever the lookup
    order happens to favour, silently changing the model's equations.
    Spec §26.10: such names are an error, "rather than guessing".

    This pass enumerates every declared identifier — BOTH the base name
    AND its fully-expanded/stratified names (reviewer feedback: a
    base-name-only check leaves a residual hole on expanded names, e.g. a
    compartment `R0` stratified to `R0_a` colliding with indexed param
    `R0[g]` expanding to `R0_a`) — across the namespaces that
    `resolve_ident_name` consults (compartments, parameters, lets,
    forcings, tables). Any name claimed by two declarations is a hard
    E278, naming both declarations and their source locations where the
    AST retains them (compartments/params carry locs; lets/tables/forcings
    do not, so those are named without a span). *)
let check_declaration_names ctx =
  (* Each entry: (identifier, namespace-label, source-loc).  We record one
     entry per *occurrence*; a name with >1 entry is the collision. *)
  let entries : (string * string * Diagnostics.loc) list ref = ref [] in
  let add name ns loc = entries := (name, ns, loc) :: !entries in
  (* Compartments — the stratified cells, plus (for a stratified compartment)
     the bare base name: a stratified `R` answers to `R` (a PopSum aggregate) as
     well as `R_north`/`R_south`, so a `let`/param sharing the base must collide.
     Registering only the cells let the base silently shadow the aggregate. *)
  List.iter (fun (cd : compartment_decl) ->
    let loc = diag_loc_of_ast_ctx ctx cd.cloc in
    let cells = expand_compartment_name ctx cd.cname in
    let names =
      if comp_dims ctx cd.cname = [] then cells else cd.cname :: cells in
    List.iter (fun n -> add n "compartment" loc) names
  ) ctx.comp_decls;
  (* Parameters — scalar by name; indexed by expanded `<base>_<level>`. *)
  List.iter (fun pd ->
    match pd with
    | PScalar p -> add p.pname "parameter" (diag_loc_of_ast_ctx ctx p.ploc)
    | PIndexed p ->
      let loc = diag_loc_of_ast_ctx ctx p.ploc in
      List.iter (fun n -> add n "parameter" loc)
        (expand_indexed_decl_names ctx p.pname p.pdims)
  ) ctx.param_decls;
  (* Let bindings — by name (no source loc in the AST). *)
  List.iter (fun lb -> add lb.lname "let" Diagnostics.no_loc) ctx.let_bindings;
  (* Forcings / time functions — base + expanded over declared indices. *)
  List.iter (fun (fd : func_decl) ->
    let dims = List.filter_map (function
      | IBind (_, d) | IConsec (_, _, d) -> Some d | IComp _ -> None) fd.findices in
    List.iter (fun n -> add n "forcing" Diagnostics.no_loc)
      (expand_indexed_decl_names ctx fd.fname dims)
  ) ctx.func_decls;
  (* Tables — by each declared name (multi-value `read` declares several). *)
  List.iter (fun (td : table_decl) ->
    List.iter (fun n -> add n "table" Diagnostics.no_loc) td.tnames
  ) ctx.table_decls;
  (* Group occurrences by identifier, preserving first-seen order for
     deterministic diagnostics. *)
  let order = ref [] in
  let groups : (string, (string * Diagnostics.loc) list ref) Hashtbl.t =
    Hashtbl.create 64 in
  List.iter (fun (name, ns, loc) ->
    match Hashtbl.find_opt groups name with
    | Some r -> r := (ns, loc) :: !r
    | None -> Hashtbl.add groups name (ref [(ns, loc)]); order := name :: !order
  ) (List.rev !entries);
  List.iter (fun name ->
    let occs = List.rev !(Hashtbl.find groups name) in
    if List.length occs > 1 then begin
      let nss = List.sort_uniq compare (List.map fst occs) in
      let message =
        if List.length nss = 1 then
          Printf.sprintf
            "duplicate %s declaration '%s': declared %d times"
            (List.hd nss) name (List.length occs)
        else
          Printf.sprintf
            "name '%s' is declared in multiple namespaces (%s); a \
             reference to it would be ambiguous"
            name (String.concat ", " nss)
      in
      let related = List.map (fun (ns, (loc : Diagnostics.loc)) ->
        if loc.Diagnostics.line > 0
        then (loc, Printf.sprintf "declared here as %s" ns)
        else (Diagnostics.no_loc,
              Printf.sprintf "also declared as %s '%s'" ns name)
      ) occs in
      (* Point the primary span at the first occurrence that has a real loc. *)
      let primary = match List.find_opt (fun (_, (l : Diagnostics.loc)) ->
          l.Diagnostics.line > 0) occs with
        | Some (_, l) -> l | None -> Diagnostics.no_loc in
      Diagnostics.error ctx.diags ~code:"E278" ~loc:primary ~message
        ~hint:"declaration names must be unique across compartments, \
               parameters, lets, forcings, and tables (including after \
               stratification expansion) — rename or remove one"
        ~related ()
    end
  ) (List.rev !order)

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

(* ── Compile-time integer evaluator (for add_calendar_* and date_range) ───── *)

(** Try to const-evaluate an expression as an integer. Accepts plain
    `EConst` (if integer-valued), `EUnOp(Neg, ...)`, and arithmetic
    of integers. Returns [None] if non-constant or non-integer.
    Phase 2 of the 2026-05-22 typed-time proposal §4: `n` in
    `add_calendar_months(d, n)` and `count`/`calendar_months`/
    `calendar_years` in `date_range` must be compile-time integers. *)
let rec try_eval_const_int (e : expr) : int option =
  match e with
  | EConst f when Float.is_integer f && Float.abs f < 1e15 ->
    Some (int_of_float f)
  | EUnOp (Neg, a) ->
    (match try_eval_const_int a with
     | Some i -> Some (-i)
     | None -> None)
  | EBinOp (op, l, r) ->
    (match try_eval_const_int l, try_eval_const_int r with
     | Some a, Some b ->
       (match op with
        | Add -> Some (a + b)
        | Sub -> Some (a - b)
        | Mul -> Some (a * b)
        | _ -> None)
     | _ -> None)
  | _ -> None

(** Try to const-evaluate an expression as an Instant — a compile-time
    (y, m, d) calendar triple. Admissible forms per proposal §4:

    - `date("YYYY-MM-DD")` literal,
    - `origin` identifier (resolves via [ctx.origin]; only in
      anchored mode),
    - `add_calendar_months(d, n)` or `add_calendar_years(d, n)`
      nested call (recurses on `d`).

    Returns [Ok (y,m,d)] on success, [Error msg] on a shape error,
    or [Error "_unanchored"] as a sentinel when an `origin`
    reference is hit in unanchored mode (caller turns this into the
    proper E327). *)
let rec try_eval_const_instant_ymd ctx (e : expr) : (int * int * int, string) result =
  match e with
  | EFuncCall ("date", [("", EIdent (s, _))]) ->
    parse_iso_date s
  | EIdent ("origin", _) ->
    (match ctx.origin with
     | Some s -> parse_iso_date s
     | None -> Error "_unanchored")
  | EFuncCall ("add_calendar_months", args) ->
    (match args with
     | [("", d); ("", n)] ->
       (match try_eval_const_instant_ymd ctx d, try_eval_const_int n with
        | Ok ymd, Some k -> Ok (add_calendar_months_ymd ymd k)
        | Error e, _ -> Error e
        | Ok _, None ->
          Error "add_calendar_months: second argument must be a compile-time integer")
     | _ -> Error "add_calendar_months: expected (Instant, Int) positional arguments")
  | EFuncCall ("add_calendar_years", args) ->
    (match args with
     | [("", d); ("", n)] ->
       (match try_eval_const_instant_ymd ctx d, try_eval_const_int n with
        | Ok ymd, Some k -> Ok (add_calendar_years_ymd ymd k)
        | Error e, _ -> Error e
        | Ok _, None ->
          Error "add_calendar_years: second argument must be a compile-time integer")
     | _ -> Error "add_calendar_years: expected (Instant, Int) positional arguments")
  | _ ->
    Error "expected a compile-time-constant date: a `date(\"...\")` literal, \
           `origin`, or a nested `add_calendar_months` / `add_calendar_years` call"

(** Detect the literal nested round-trip pattern
    `add_calendar_months(add_calendar_months(d, n), -n)` (and same
    for `add_calendar_years`). Single-shape syntactic match per
    proposal §4: let-laundered cases do not trigger.

    Returns [Some primitive_name] iff the outer call is a round-trip
    over the inner call with negated `n`. *)
let detect_round_trip ~primitive_name ~outer_n ~inner =
  match inner with
  | EFuncCall (fn, [("", _); ("", inner_n_expr)]) when fn = primitive_name ->
    (match try_eval_const_int inner_n_expr with
     | Some inner_n -> inner_n = - outer_n && inner_n <> 0
     | None -> false)
  | _ -> false

(** Expand a `date_range(...)` AST call to a list of `EConst` exprs
    in the model's time units (days-from-origin). Phase 2 of the
    2026-05-22 typed-time proposal §4.

    Forms accepted (exactly one cadence kwarg):
    - `date_range(start, end, every = D)` — affine
    - `date_range(start, count = N, every = D)` — affine count
    - `date_range(start, end, calendar_months = N)` — calendar
    - `date_range(start, count = N, calendar_months = N)` — calendar count
    - same with `calendar_years` instead of `calendar_months`.

    `inclusive_end : Bool = true` defaults to true.

    Errors and warnings are emitted on [ctx.diags] and the partial
    list is returned (compilation continues to collect more diags).
    Returns a list of `EConst` AST exprs that the caller can splice
    in-place where a list of constant time expressions is expected. *)
let expand_date_range_to_consts ctx (args : (string * expr) list) : expr list =
  let origin_set = ctx.origin <> None in
  let get_kw k = List.assoc_opt k args in
  let positional = List.filter_map
    (fun (k, e) -> if k = "" then Some e else None) args in
  let err code msg ?hint () =
    Diagnostics.error ctx.diags ~code ~loc:Diagnostics.no_loc
      ~message:msg ?hint ()
  in
  let warn code msg ?hint () =
    Diagnostics.warning ctx.diags ~code ~loc:Diagnostics.no_loc
      ~message:msg ?hint ()
  in
  (* Resolve `start` from the first positional arg. *)
  let start_expr = match positional with
    | s :: _ -> Some s
    | [] ->
      err "E328"
        "date_range: missing `start` (first positional argument)"
        ~hint:"example: date_range(date(\"2020-01-01\"), \
               date(\"2020-12-31\"), every = 7 'days)" ();
      None
  in
  (* `end` is the optional second positional. If absent, `count =`
     must be supplied. *)
  let end_expr = match positional with
    | _ :: e :: _ -> Some e
    | _ -> None
  in
  let count_kw = get_kw "count" in
  (* Cadence: exactly one of `every`, `calendar_months`,
     `calendar_years` must be present. *)
  let every_kw   = get_kw "every"           in
  let cmonths_kw = get_kw "calendar_months" in
  let cyears_kw  = get_kw "calendar_years"  in
  let cadence_present =
    List.filter Option.is_some [every_kw; cmonths_kw; cyears_kw]
    |> List.length
  in
  if cadence_present = 0 then begin
    err "E328"
      "date_range: missing cadence kwarg — supply exactly one of \
       `every = D`, `calendar_months = N`, or `calendar_years = N`"
      ~hint:"example: date_range(date(\"2020-01-01\"), \
             date(\"2020-12-31\"), every = 7 'days)" ();
    []
  end
  else if cadence_present > 1 then begin
    err "E328"
      "date_range: only one cadence kwarg allowed — `every`, \
       `calendar_months`, and `calendar_years` are mutually exclusive"
      () ;
    []
  end
  else
  (* end/count: exactly one. *)
  let _ =
    match end_expr, count_kw with
    | Some _, Some _ ->
      err "E328"
        "date_range: `end` (second positional) and `count = ...` are \
         mutually exclusive" ();
    | None, None ->
      err "E328"
        "date_range: needs either an `end` positional argument or a \
         `count = N` kwarg"
        ~hint:"example: date_range(date(\"2020-01-01\"), \
               count = 24, every = 7 'days)" ();
    | _ -> ()
  in
  let inclusive_end =
    match get_kw "inclusive_end" with
    | None -> true
    | Some (EIdent ("true", _))  | Some (EFuncCall ("true", []))  -> true
    | Some (EIdent ("false", _)) | Some (EFuncCall ("false", [])) -> false
    | Some _ ->
      err "E328"
        "date_range: `inclusive_end = ...` must be `true` or `false`" ();
      true
  in
  let validate_kwargs () =
    let known = ["count"; "every"; "calendar_months"; "calendar_years";
                 "inclusive_end"] in
    List.iter (fun (k, _) ->
      if k <> "" && not (List.mem k known) then
        err "E328"
          (Printf.sprintf "date_range: unknown keyword argument `%s`" k)
          ~hint:(Printf.sprintf "valid kwargs: %s" (String.concat ", " known))
          ()
    ) args
  in
  validate_kwargs ();
  (* Resolve start to (y, m, d). *)
  let start_ymd = match start_expr with
    | None -> None
    | Some e ->
      (match try_eval_const_instant_ymd ctx e with
       | Ok ymd -> Some ymd
       | Error "_unanchored" ->
         err "E327"
           "`date_range` with `start = origin` requires an anchored \
            model"
           ~hint:"add `origin = date(\"YYYY-MM-DD\")` at the top of \
                  the file" ();
         None
       | Error msg ->
         err "E328" (Printf.sprintf "date_range: %s" msg)
           ~hint:"`start` must be a `date(\"...\")` literal, `origin`, \
                  or an `add_calendar_*` call" ();
         None)
  in
  (* Resolve optional end to (y, m, d). *)
  let end_ymd = match end_expr with
    | None -> None
    | Some e ->
      (match try_eval_const_instant_ymd ctx e with
       | Ok ymd -> Some ymd
       | Error "_unanchored" -> None
       | Error msg ->
         err "E328" (Printf.sprintf "date_range: %s" msg) ();
         None)
  in
  (* Resolve count if present, must be positive. *)
  let count_int = match count_kw with
    | None -> None
    | Some e ->
      (match try_eval_const_int e with
       | Some n when n >= 1 -> Some n
       | Some n ->
         err "E329"
           (Printf.sprintf
             "date_range: `count = %d` must be ≥ 1" n) ();
         None
       | None ->
         err "E328"
           "date_range: `count` must be a compile-time integer" ();
         None)
  in
  (* Convert a (y,m,d) to a model-time-unit float, days-from-origin. *)
  let ymd_to_float ymd =
    match ctx.origin with
    | Some origin_str ->
      let iso = format_iso_date ymd in
      (try parse_date_to_float origin_str iso ctx.time_unit
       with Failure msg ->
         (* M13 (gh#98): no silent `0.0` absorb. A failure here means the
            `origin` string is malformed (the generated `iso` always
            parses); surface it as a located E223 instead of computing a
            garbage day offset. *)
         Diagnostics.error ctx.diags ~code:"E223" ~loc:Diagnostics.no_loc
           ~message:msg ();
         0.0)
    | None ->
      (* Unanchored: dates are not meaningful relative to origin.
         We already errored upstream; return 0 defensively. *)
      0.0
  in
  let cmp_ymd (y1, m1, d1) (y2, m2, d2) =
    compare (y1, m1, d1) (y2, m2, d2)
  in
  let leq_ymd a b = cmp_ymd a b <= 0 in
  let _ = leq_ymd in
  (* Now branch on cadence flavor. *)
  match start_ymd with
  | None -> []
  | Some s_ymd ->
    let entries_ymd =
      if Option.is_some every_kw then begin
        (* Affine cadence. `every` is a duration expr in 'days or
           'weeks. We need its value in DAYS (calendar arithmetic
           lives in days, not in model time units). *)
        let every_e = Option.get every_kw in
        let days_per_unit = function
          | Days | PerDay -> Some 1.0
          | Weeks | PerWeek -> Some 7.0
          | _ -> None
        in
        let every_days = match every_e with
          | EUnit (f, u) -> (match days_per_unit u with
              | Some k -> Some (f *. k)
              | None ->
                err "E328"
                  (Printf.sprintf
                    "date_range: `every` must be in `'days` or `'weeks` \
                     (got `'%s`)" (unit_lit_to_string u))
                  ~hint:"calendar cadences use `calendar_months = N` or \
                         `calendar_years = N` instead"
                  ();
                None)
          | _ ->
            err "E328"
              "date_range: `every` must be a duration literal with \
               `'days` or `'weeks` (e.g. `every = 7 'days`)" ();
            None
        in
        match every_days with
        | None -> []
        | Some d when d <= 0.0 ->
          err "E329"
            (Printf.sprintf
              "date_range: `every` must be positive (got %g)" d) ();
          []
        | Some step_days_f ->
          (* Step the affine cadence in proleptic-Gregorian day
             indices. Convert start to absolute day-of-epoch via
             [days_of_date], step, and convert back. We round
             step_days_f to an integer (sub-day steps are not
             supported per docs/dates.md). *)
          let step_days = int_of_float (Float.round step_days_f) in
          if step_days <= 0 then begin
            err "E329"
              "date_range: `every` rounds to zero days — must be ≥ 1 day" ();
            []
          end else
            let (sy, sm, sd) = s_ymd in
            let start_dn = days_of_date sy sm sd in
            (* Convert a day-number back to (y,m,d) by linear search
               anchored at start: incrementally add step_days. We
               don't need a general inverse — we just step forward
               from a known (y,m,d). *)
            let step_ymd ymd k_days =
              (* Add k_days to (y,m,d) by walking days_in_month. *)
              let (y, m, d) = ymd in
              let rec walk y m d k =
                if k = 0 then (y, m, d)
                else if k > 0 then
                  let dim = days_in_month y m in
                  if d + k <= dim then (y, m, d + k)
                  else
                    let remaining = k - (dim - d + 1) in
                    let y', m' =
                      if m = 12 then (y + 1, 1) else (y, m + 1)
                    in
                    walk y' m' 1 remaining
                else
                  (* k < 0 *)
                  if d + k >= 1 then (y, m, d + k)
                  else
                    let y', m' =
                      if m = 1 then (y - 1, 12) else (y, m - 1)
                    in
                    let dim' = days_in_month y' m' in
                    walk y' m' dim' (k + d)
              in
              walk y m d k_days
            in
            (* Generate boundaries. *)
            let entries = ref [s_ymd] in
            (match end_ymd, count_int with
             | _, Some n ->
               (* Count form: produces count + 1 entries. *)
               let cur = ref s_ymd in
               for _ = 1 to n do
                 cur := step_ymd !cur step_days;
                 entries := !cur :: !entries
               done;
               List.rev !entries
             | Some e_ymd, None ->
               (* Start–end form. Step until > end. *)
               let (ey, em, ed) = e_ymd in
               let end_dn = days_of_date ey em ed in
               let cur = ref s_ymd in
               let cur_dn = ref start_dn in
               let aligned = ref true in
               while !cur_dn + step_days <= end_dn do
                 cur := step_ymd !cur step_days;
                 cur_dn := !cur_dn + step_days;
                 entries := !cur :: !entries
               done;
               if !cur_dn <> end_dn then aligned := false;
               (* inclusive_end semantics: end is appended ONLY when
                  it equals a boundary; otherwise W328 fires. *)
               if (not !aligned) && inclusive_end then begin
                 warn "W328"
                   (Printf.sprintf
                     "date_range: `end = %s` does not land on a cadence \
                      boundary from `start = %s`; last produced entry \
                      is %s"
                     (format_iso_date e_ymd)
                     (format_iso_date s_ymd)
                     (format_iso_date !cur))
                   ~hint:"list `end` explicitly or set \
                          `inclusive_end = false` to make the \
                          truncation explicit"
                   ()
               end;
               List.rev !entries
             | None, None -> [s_ymd])
      end
      else begin
        (* Calendar cadence: months or years. *)
        if not origin_set then begin
          let kw = if Option.is_some cmonths_kw
            then "calendar_months" else "calendar_years" in
          err "E327"
            (Printf.sprintf
              "date_range with `%s` cadence requires an anchored model" kw)
            ~hint:"add `origin = date(\"YYYY-MM-DD\")` at the top of \
                   the file, or use `every = N 'days` for an affine \
                   cadence"
            ();
          []
        end else
          let n_step, step_fn =
            if Option.is_some cmonths_kw then
              (Option.get cmonths_kw, add_calendar_months_ymd)
            else
              (Option.get cyears_kw, add_calendar_years_ymd)
          in
          let k_step = match try_eval_const_int n_step with
            | Some n -> n
            | None ->
              err "E328"
                "date_range: calendar cadence must be a \
                 compile-time integer" ();
              0
          in
          if k_step <= 0 then begin
            err "E329"
              (Printf.sprintf
                "date_range: calendar cadence must be positive (got %d)"
                k_step) ();
            []
          end else
            let entries = ref [s_ymd] in
            (match end_ymd, count_int with
             | _, Some n ->
               let cur = ref s_ymd in
               for _ = 1 to n do
                 cur := step_fn !cur k_step;
                 entries := !cur :: !entries
               done;
               List.rev !entries
             | Some e_ymd, None ->
               (* Step until next would exceed end. *)
               let cur = ref s_ymd in
               let stop = ref false in
               while not !stop do
                 let nxt = step_fn !cur k_step in
                 if cmp_ymd nxt e_ymd <= 0 then begin
                   cur := nxt;
                   entries := nxt :: !entries
                 end else
                   stop := true
               done;
               if cmp_ymd !cur e_ymd <> 0 && inclusive_end then
                 warn "W328"
                   (Printf.sprintf
                     "date_range: `end = %s` does not land on a cadence \
                      boundary from `start = %s`; last produced entry \
                      is %s"
                     (format_iso_date e_ymd)
                     (format_iso_date s_ymd)
                     (format_iso_date !cur))
                   ~hint:"list `end` explicitly or set \
                          `inclusive_end = false` to make the \
                          truncation explicit"
                   ();
               List.rev !entries
             | None, None -> [s_ymd])
      end
    in
    List.map (fun ymd -> EConst (ymd_to_float ymd)) entries_ymd

(** Splice any `date_range(...)` calls in a list of AST exprs
    in-place, producing a flat list with all date_ranges materialized
    to their `EConst` entries. Used at list-consuming sites
    (`at = [...]`, `on = [...]`, table inline values). *)
let splice_date_ranges ctx (es : expr list) : expr list =
  List.concat_map (fun e ->
    match e with
    | EFuncCall ("date_range", args) -> expand_date_range_to_consts ctx args
    | _ -> [e]
  ) es

(* ── Fix B: shared-binding extraction ─────────────────────────────────────── *)

(* A reference inside a `let` body that makes it ineligible for hoisting: a
   parameter (the body would have d/dp ≠ 0, but the BindingRef autodiff arm
   yields 0 — a silent zero gradient) or another `let` (conservatively
   excluded; it may transitively carry a parameter). Compartments, tables,
   forcings, time, and constants are all fine — param-free, so d/dp ≡ 0. *)
let rec body_refs_param_or_let ctx (e : expr) : bool =
  let bad n =
    is_indexed_param ctx n
    || Hashtbl.mem ctx.scalar_param_tbl n
    || Hashtbl.mem ctx.let_tbl n
  in
  match e with
  | EConst _ | EUnit _ -> false
  | EIdent (n, _)      -> bad n
  | EIndex (n, items, _)  ->
    bad n
    || List.exists (function IPosn e | INamed (_, e) -> body_refs_param_or_let ctx e) items
  | EBinOp (_, l, r) -> body_refs_param_or_let ctx l || body_refs_param_or_let ctx r
  | EUnOp (_, e)     -> body_refs_param_or_let ctx e
  | ESum (_, _, _, b) -> body_refs_param_or_let ctx b
  | ECond (p, t, f)  ->
    body_refs_param_or_let ctx p || body_refs_param_or_let ctx t || body_refs_param_or_let ctx f
  | EFuncCall (_, args) -> List.exists (fun (_, e) -> body_refs_param_or_let ctx e) args
  | EList es            -> List.exists (body_refs_param_or_let ctx) es
  | ERange (lo, hi)     -> body_refs_param_or_let ctx lo || body_refs_param_or_let ctx hi
  | EObsAccess _        -> false
  | ERunMember _        -> false

(* Every index-position variable in the body must be bound by the let's own
   declared indices or an enclosing `sum`; otherwise the resolved body depends
   on the enclosing transition's indices (the `inner_env @ env` join) and is
   not a context-independent shared value. Literal dimension levels in index
   position are conservatively rejected (treated as unbound) — none of the
   extractable per-coordinate aggregates use them. *)
let free_index_var_clean (lb : let_binding) : bool =
  let declared = List.concat_map (function
    | IBind (v, _)       -> [v]
    | IConsec (v, vn, _) -> [v; vn]
    | IComp v            -> [v]) lb.lindices in
  let rec ok bound (e : expr) = match e with
    | EConst _ | EUnit _ | EIdent _ -> true
    | EIndex (_, items, _) ->
      List.for_all (function
        | IPosn (EIdent (v, _)) | INamed (_, EIdent (v, _)) -> List.mem v bound
        | IPosn e | INamed (_, e) -> ok bound e) items
    | EBinOp (_, l, r) -> ok bound l && ok bound r
    | EUnOp (_, e)     -> ok bound e
    | ESum (v, _, _, b) -> ok (v :: bound) b
    | ECond (p, t, f)  -> ok bound p && ok bound t && ok bound f
    | EFuncCall (_, args) -> List.for_all (fun (_, e) -> ok bound e) args
    | EList es            -> List.for_all (ok bound) es
    | ERange (lo, hi)     -> ok bound lo && ok bound hi
    | EObsAccess _        -> true
    | ERunMember _        -> true
  in
  ok declared lb.lbody

(* Memoized hoist-eligibility for a `let` (by name). Eligible ⇒ extract once
   into `model.bindings`; ineligible ⇒ keep inlining (the prior behaviour). *)
let let_is_hoistable ctx (lb : let_binding) : bool =
  match Hashtbl.find_opt ctx.hoist_memo lb.lname with
  | Some b -> b
  | None ->
    let eligible =
      lb.lshape = None
      && not (lb.lkind <> None && is_const_expr lb.lbody)   (* typed const → Param path *)
      && not (body_refs_param_or_let ctx lb.lbody)
      && free_index_var_clean lb
    in
    Hashtbl.replace ctx.hoist_memo lb.lname eligible;
    eligible

(* Register a hoisted binding under its concrete (index-mangled) name, once,
   and return the `BindingRef` to substitute at the use site. The body thunk
   is resolved on first registration only; because it resolves nested
   let-uses first (registering THEIR bindings before this one is prepended),
   `hoisted_rev` is reverse-topological and `collect_hoisted_bindings`
   reverses it so each binding's dependencies precede it. *)
let register_hoisted_binding ctx (concrete : string) (resolve_body : unit -> Ir.expr) : Ir.expr =
  if not (Hashtbl.mem ctx.hoisted_tbl concrete) then begin
    Hashtbl.replace ctx.hoisted_tbl concrete ();
    let body = resolve_body () in
    ctx.hoisted_rev <- (concrete, body) :: ctx.hoisted_rev
  end;
  Ir.BindingRef concrete

let collect_hoisted_bindings ctx : Ir.binding list =
  List.rev_map (fun (name, body) -> { Ir.bname = name; Ir.bexpr = body }) ctx.hoisted_rev

(* ── Guard evaluation ─────────────────────────────────────────────────────── *)
(* Defined before [resolve_expr] because the restricted-sum `where` filter calls
   [eval_guard] during expression resolution (gh#185). *)

let apply_relop op (a : float) (b : float) : bool =
  match op with
  | RLt -> a < b  | RLe -> a <= b
  | RGt -> a > b  | RGe -> a >= b
  | REq -> a = b  | RNe -> a <> b

(* Resolve the constant value of a table cell referenced by a `where`
   predicate: table name + index variables (resolved to dimension levels via
   [env], row-major flattened offset). Reads the pre-built [ctx.table_index].
   Emits E284 (predicate not compile-time decidable) and returns None on any
   failure — unknown/non-constant table, bad arity, unknown level, OOB, or a
   non-constant (parameterized) cell. *)
(* Row-major flat offset of a table cell: per-dimension positions (0-based) and
   the dimension sizes, folded Horner-style (dimension 0 varies slowest). Shared
   by the where-predicate cell reader and the gh#345 table-backed forcing slice
   so the stride math lives in one place. *)
let row_major_offset positions sizes =
  List.fold_left2 (fun acc pos sz -> (acc * sz) + pos) 0 positions sizes

let eval_tab_cell ctx env tname idxs : float option =
  let err msg =
    Diagnostics.error ctx.diags ~code:"E284" ~loc:Diagnostics.no_loc ~message:msg ();
    None
  in
  match Hashtbl.find_opt ctx.table_index tname with
  | None ->
    err (Printf.sprintf
      "the where-predicate references '%s', which is not a compile-time-constant \
       table; a `where` predicate must be decidable before simulation (it may \
       reference index variables and constant tables only)." tname)
  | Some (cells, dims) ->
    if List.length idxs <> List.length dims then
      err (Printf.sprintf
        "the where-predicate indexes table '%s' with %d indices, but it has %d \
         dimensions." tname (List.length idxs) (List.length dims))
    else begin
      let levels_of dim = match List.assoc_opt dim ctx.dim_registry with
        | Some l -> l | None -> [] in
      let positions = List.map2 (fun var dim ->
        let lvl = match List.assoc_opt var env with Some l -> l | None -> var in
        List.find_index (fun v -> v = lvl) (levels_of dim)
      ) idxs dims in
      let sizes = List.map (fun dim -> List.length (levels_of dim)) dims in
      if List.exists Option.is_none positions then
        err (Printf.sprintf
          "the where-predicate indexes table '%s' with a value that is not a \
           known level of its dimension." tname)
      else
        let positions = List.map Option.get positions in
        let offset = row_major_offset positions sizes in
        if offset < 0 || offset >= Array.length cells then
          err (Printf.sprintf "the where-predicate index into table '%s' is out of bounds." tname)
        else match cells.(offset) with
          | Ir.Const f -> Some f
          | _ ->
            err (Printf.sprintf
              "the where-predicate references a non-constant (parameterized) cell of \
               table '%s'; the support must be a compile-time constant. Use a \
               constant mask/distance table in the predicate and keep fitted \
               weights in the rate body." tname)
    end

let rec eval_guard ctx env = function
  | GEq (a, b) ->
    let va = Option.value ~default:a (List.assoc_opt a env) in
    let vb = Option.value ~default:b (List.assoc_opt b env) in
    va = vb
  | GNeq (a, b) ->
    let va = Option.value ~default:a (List.assoc_opt a env) in
    let vb = Option.value ~default:b (List.assoc_opt b env) in
    va <> vb
  | GTab (tname, idxs, op, operand) ->
    (match operand with
     | GoName n ->
       (* A name as the threshold. camdl has no compile-time-constant scalars,
          so it is a parameter (or unknown): a fitted radius would change which
          patches couple at runtime — an unbounded reduction. Targeted error. *)
       Diagnostics.error ctx.diags ~code:"E284" ~loc:Diagnostics.no_loc
         ~message:(Printf.sprintf
           "the where-predicate compares table '%s' against '%s', but a coupling \
            support must be fixed at compile time. If '%s' is a parameter, a \
            fitted threshold would change which patches couple at runtime (an \
            unbounded reduction the engine cannot evaluate). Use a literal \
            threshold (e.g. `< 50`) for the support and fit the kernel's \
            shape/strength in the rate body instead." tname n n) ();
       false
     | GoNum rhs ->
       (match eval_tab_cell ctx env tname idxs with
        | Some cell -> apply_relop op cell rhs
        | None      -> false))   (* error already emitted; drop the term *)
  | GAnd (g1, g2) -> eval_guard ctx env g1 && eval_guard ctx env g2
  | GOr  (g1, g2) -> eval_guard ctx env g1 || eval_guard ctx env g2

(* v1 generated-quantity temporal-reduction names (proposal 2026-06-25). These
   are NOT lexer keywords — they lex as IDENT and dispatch by name in the
   quantity classifier. `max`/`min` are deliberately ABSENT: they stay binary
   pointwise operators everywhere (resolve_expr lowers a 2-arg `max`/`min` to
   Ir.BinOp Max/Min); a UNARY `max`/`min` is intercepted as a reduction inside
   the classifier only. The classifier dispatches on this set; resolve_expr uses
   it to reject a reduction name that leaked into a rate / binding (E290). *)
let temporal_reduction_names =
  [ "final"; "mean"; "integral";
    "count_above"; "count_below";
    "time_of_max"; "time_of_min";
    "first_above"; "first_below"; "last_above"; "last_below" ]

let is_temporal_reduction_name n = List.mem n temporal_reduction_names

(* Declared index dims of an indexed parameter (`R0[patch]` → ["patch"]), or
   None for a scalar / unknown name. *)
let indexed_param_dims ctx name =
  List.find_map (function
    | PIndexed p when p.pname = name -> Some p.pdims
    | _ -> None) ctx.param_decls

(* Shared arity check for an indexed reference `Name[i, j, ...]`. The table
   (E202), shaped-let (E273), and compartment (E287) lookups keep their own
   guards; this is the one check for the let / forcing / parameter branches,
   which previously dropped (let) or name-mangled (forcing/param) a mismatched
   index count. Returns true iff the arity matches; on mismatch it emits a
   located E299 and the caller substitutes a placeholder so the pass keeps
   collecting diagnostics. *)
let check_index_arity ctx ~loc ~kind ~name ~declared ~provided : bool =
  if declared = provided then true
  else begin
    let plural n = if n = 1 then "index" else "indices" in
    Diagnostics.error ctx.diags
      ~code:"E299" ~loc
      ~message:(Printf.sprintf "%s '%s' expects %d %s but was given %d"
                  kind name declared (plural declared) provided)
      ~hint:(Printf.sprintf "index it with exactly %d %s" declared (plural declared))
      ();
    false
  end

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
  | EIndex (name, items, l) -> (
    let idx_loc = diag_loc_of_ast_ctx ctx l in
    let base_name =
      match List.assoc_opt name env with Some n -> n | None -> name
    in
    (* 1. Table? → TableLookup with a single flattened linear index.
       For a table of dims [d1; d2; ...] with sizes [n1; n2; ...], the
       linear index is: i1*n2*n3*... + i2*n3*... + ... + iN.
       The IR and Rust runtime always expect exactly one index. *)
    let tdims = table_dims ctx base_name in
    if tdims <> [] && List.length items <> List.length tdims then begin
      (* gh#112: table-lookup arity guard. The stride math below maps over
         the user's `items`, NOT the declared `tdims` — so an under-indexed
         lookup (`C_age[a]` against `age × age`) silently produced a
         partial-prefix linear index (a wrong cell), and an over-indexed one
         read `tdims` out of range. Require exact arity before lowering.
         Mirrors shape_index's E273 guard for shaped lets. *)
      Diagnostics.error ctx.diags
        ~code:"E202" ~loc:Diagnostics.no_loc
        ~message:(Printf.sprintf
          "table '%s' expects %d %s but was given %d"
          base_name (List.length tdims)
          (if List.length tdims = 1 then "index" else "indices")
          (List.length items))
        ~hint:(Printf.sprintf "table dimensions: [%s]"
          (String.concat " \xc3\x97 " tdims))
        ();
      (* Continue with a placeholder so a single pass collects further
         diagnostics; the compile aborts at phase end. *)
      Ir.TableLookup (base_name, [Ir.Const 0.0])
    end
    else if tdims <> [] then
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
      if not (check_index_arity ctx ~loc:idx_loc ~kind:"let binding" ~name:base_name
                ~declared:(List.length lb.lindices) ~provided:(List.length items))
      then Ir.Const 0.0
      else
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
      (* Fix B: extract a state-only, context-independent per-coordinate let
         (e.g. N[l], I_lga[l]) into a shared binding once instead of inlining
         its body at every use. Eligible bodies resolve against inner_env
         alone (no enclosing-transition dependence). Ineligible lets inline
         exactly as before. *)
      if let_is_hoistable ctx lb && not ctx.suppress_hoist then
        let concrete = String.concat "_" (base_name :: List.map snd inner_env) in
        register_hoisted_binding ctx concrete
          (fun () -> normalize_expr (resolve_expr ctx inner_env lb.lbody))
      else
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
      (let fd = Hashtbl.find ctx.func_tbl base_name in
       if not (check_index_arity ctx ~loc:idx_loc ~kind:"forcing" ~name:base_name
                 ~declared:(List.length fd.findices) ~provided:(List.length items))
       then Ir.Const 0.0
       else
         let idx_vals = List.map (index_item_to_str env) items in
         Ir.TimeFunc (String.concat "_" (base_name :: idx_vals)))
    else
    (* 3. Indexed parameter? → resolve index and return Ir.Param of mangled name *)
    if is_indexed_param ctx base_name then
      (let declared =
         match indexed_param_dims ctx base_name with Some d -> List.length d | None -> 1 in
       if not (check_index_arity ctx ~loc:idx_loc ~kind:"parameter" ~name:base_name
                 ~declared ~provided:(List.length items))
       then Ir.Const 0.0
       else
       match items with
       | [IPosn (EIdent (idx, _))] | [INamed (_, EIdent (idx, _))] ->
         let concrete = resolve_index ctx env idx in
         Ir.Param (base_name ^ "_" ^ concrete)
       | _ ->
         (* multi-item or non-ident index: fall through to name mangling *)
         let idx_vals = List.map (index_item_to_str env) items in
         let concrete = String.concat "_" (base_name :: idx_vals) in
         resolve_ident_name ctx concrete ~loc:Diagnostics.no_loc)
    else
    (* 4. Compartment with indices → concatenate to concrete name.
       Partial index guard (E287): a compartment stratified over 2+ dimensions
       referenced with *some but not all* dimensions dropped (e.g. `E[a]` when
       `E` has `[age, latent_stage]`) has no well-defined cell. Omitting *all*
       dimensions (the bare name `E`) sums over them — that path is handled in
       `resolve_ident_name` and never reaches here. But a partial index used to
       fall through to name-mangling (`E_adult`), then E100'd against a synthetic
       compartment the user never wrote — no source loc, a name they can't act
       on. Reject it here with a located diagnostic that names the real
       compartment, its dimensions, and the explicit-marginalization fix. *)
    let comp_dim_list = comp_dims ctx base_name in
    let n_dims  = List.length comp_dim_list in
    let n_items = List.length items in
    if Hashtbl.mem ctx.comp_tbl base_name && n_items > 0 && n_items < n_dims then begin
      let dims_str = String.concat ", " comp_dim_list in
      (* the dropped dimensions are the suffix beyond what the user indexed *)
      let dropped = List.filteri (fun i _ -> i >= n_items) comp_dim_list in
      let one_dropped = match dropped with [d] -> d | _ -> List.hd comp_dim_list in
      Diagnostics.error ctx.diags
        ~code:"E287" ~loc:idx_loc
        ~message:(Printf.sprintf
          "compartment '%s' has dimensions [%s] but only %d of %d were indexed; \
           a partial index has no defined cell"
          base_name dims_str n_items n_dims)
        ~hint:(Printf.sprintf
          "index all dimensions (e.g. `%s[%s]`), or marginalize a dimension \
           explicitly with `sum(s in %s, %s[%s, s])`"
          base_name dims_str
          one_dropped base_name
          (String.concat ", " (List.filteri (fun i _ -> i < n_items) comp_dim_list)))
        ();
      Ir.Const 0.0  (* placeholder — keep collecting diagnostics this pass *)
    end
    else
    let idx_vals = List.map (index_item_to_str env) items in
    let concrete = String.concat "_" (base_name :: idx_vals) in
    resolve_ident_name ctx concrete ~loc:idx_loc
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
  | ESum (v, d, guard_opt, body) ->
    let vals = dim_values ctx d in
    (* Restricted sum: a `where` predicate prunes the domain to the levels that
       satisfy it, evaluated at compile time (the sum var bound to each
       candidate). Survivors-only → O(P·k) by construction (gh#185). *)
    let vals = match guard_opt with
      | None   -> vals
      | Some g -> List.filter (fun vv -> eval_guard ctx ((v, vv) :: env) g) vals
    in
    if vals = [] then Ir.Const 0.0
    else
      let terms = List.map (fun vv ->
        resolve_expr ctx ((v, vv) :: env) body
      ) vals in
      (* Fix D, increment 2. If the terms are all Pop/PopSum-additive (e.g. a
         per-patch total `N = sum(a, S+E+I+R)`), build the Add-chain and let
         normalize_expr collapse it to a flat PopSum — preserving the
         IntPopSum/MixedPopSum fold order bit-for-bit (the reassociation trap:
         a source-order Reduce of a ~100-term mixed sum would flip a draw).
         Otherwise the terms are Mul-trees (the spatial coupling sum); emit a
         flat n-ary Reduce instead of a deep left-nested Add chain, which tripped
         serde's recursion limit past ~50 patches. Both forms evaluate as the
         same left-fold, so trajectories stay byte-identical (gate-verified). *)
      let add_chain =
        List.fold_left (fun acc t ->
          Ir.BinOp { op = Ir.Add; left = acc; right = t }
        ) (List.hd terms) (List.tl terms)
      in
      (match normalize_expr add_chain with
       | Ir.PopSum _ as collapsed -> collapsed
       | _ -> Ir.Reduce terms)
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
       (* Validate the date string explicitly so an out-of-range or
          malformed date gets the named E223 (gh#98, C6/M13) — not the
          generic E220, and never a silent `0.0` shift. The origin was
          validated up-front (M14), so the only remaining failure source
          here is `date_str`. *)
       (match parse_iso_date date_str with
        | Error msg ->
          Diagnostics.error ctx.diags ~code:"E223" ~loc:Diagnostics.no_loc
            ~message:msg
            ~hint:"date() takes a calendar date YYYY-MM-DD with month \
                   01..12 and a day valid for that month (leap-aware)"
            ();
          Ir.Const 0.0
        | Ok _ ->
          (try Ir.Const (parse_date_to_float origin_str date_str ctx.time_unit)
           with
           | Failure msg ->
             Diagnostics.error ctx.diags ~code:"E223" ~loc:Diagnostics.no_loc
               ~message:msg ();
             Ir.Const 0.0
           | Invalid_argument _ ->
             (* `parse_date_to_float` raises `Invalid_argument` when
                `time_unit` is not a calendar unit (`'count` / `'ratio`,
                both parseable alongside `origin = date(...)`). The data
                loader already catches this (`load_table_data`); the model
                `date()` path must too, or the bare exception escapes the
                expander as an uncaught-`E001` stack-trace (gh#134). A
                dimensionless `time_unit` has no day mapping, so a `date()`
                literal cannot be converted to internal time. *)
             Diagnostics.error ctx.diags ~code:"E223" ~loc:Diagnostics.no_loc
               ~message:(Printf.sprintf
                 "date(\"%s\") cannot be converted: `time_unit = '%s` is a \
                  dimensionless unit, so a calendar date has no day offset \
                  from `origin`"
                 date_str (unit_lit_to_string ctx.time_unit))
               ~hint:"declare a calendar `time_unit` ('days, 'weeks) to use \
                      date() literals, or write a bare numeric time instead"
               ();
             Ir.Const 0.0))
     | None ->
       Diagnostics.error ctx.diags ~code:"E220" ~loc:Diagnostics.no_loc
         ~message:"date() requires a top-level origin declaration, e.g. origin = date(\"2020-01-01\")"
         ();
       Ir.Const 0.0)
  | EFuncCall (("add_calendar_months" | "add_calendar_years") as fname, args) ->
    (* Expander-only calendar arithmetic primitives — Phase 2 of the
       2026-05-22 typed-time proposal §4. Materializes at compile
       time to `Ir.Const` (days-from-origin in `time_unit` units);
       has no runtime IR node. *)
    (match args with
     | [("", d_expr); ("", n_expr)] ->
       (* Anchored-mode gate: calendar stepping requires `origin`. *)
       (match ctx.origin with
        | None ->
          Diagnostics.error ctx.diags
            ~code:"E327" ~loc:Diagnostics.no_loc
            ~message:(Printf.sprintf
              "`%s` requires an anchored model (calendar stepping is \
               only meaningful with a calendar origin)" fname)
            ~hint:"add `origin = date(\"YYYY-MM-DD\")` at the top of \
                   the file, or use an exact-day duration like `30 'days`"
            ();
          Ir.Const 0.0
        | Some origin_str ->
          (* n must be a compile-time integer. *)
          (match try_eval_const_int n_expr with
           | None ->
             Diagnostics.error ctx.diags
               ~code:"E328" ~loc:Diagnostics.no_loc
               ~message:(Printf.sprintf
                 "%s: second argument must be a compile-time integer \
                  (number of calendar %s to add)" fname
                 (if fname = "add_calendar_months" then "months" else "years"))
               ~hint:"example: add_calendar_months(date(\"2020-01-01\"), 6)"
               ();
             Ir.Const 0.0
           | Some n ->
             (* W327 — literal round-trip composition. Fires before
                the inner is itself evaluated, so the warning lands
                even if the inner shape is legal. *)
             if detect_round_trip ~primitive_name:fname ~outer_n:n ~inner:d_expr
             then
               Diagnostics.warning ctx.diags
                 ~code:"W327" ~loc:Diagnostics.no_loc
                 ~message:(Printf.sprintf
                   "%s round-trip composition: %s(%s(d, %d), %d) is \
                    not in general equal to d — month-end clamping is \
                    non-invertible"
                   fname fname fname (-n) n)
                 ~hint:"month-end clamping is non-invertible: \
                        add_calendar_months(date(\"2020-01-31\"), 1) \
                        = date(\"2020-02-29\"), then (..., -1) = \
                        date(\"2020-01-29\"), not date(\"2020-01-31\")"
                 ();
             (* Const-evaluate the date argument. *)
             (match try_eval_const_instant_ymd ctx d_expr with
              | Ok ymd ->
                let ymd' =
                  if fname = "add_calendar_months"
                  then add_calendar_months_ymd ymd n
                  else add_calendar_years_ymd ymd n
                in
                let iso = format_iso_date ymd' in
                (try Ir.Const (parse_date_to_float origin_str iso ctx.time_unit)
                 with Failure msg ->
                   Diagnostics.error ctx.diags
                     ~code:"E328" ~loc:Diagnostics.no_loc
                     ~message:msg ();
                   Ir.Const 0.0)
              | Error "_unanchored" ->
                (* Inner `origin` reference in unanchored mode. This
                   path is unreachable in practice — the outer check
                   above already errored. Defensive. *)
                Ir.Const 0.0
              | Error msg ->
                Diagnostics.error ctx.diags
                  ~code:"E328" ~loc:Diagnostics.no_loc
                  ~message:(Printf.sprintf "%s: %s" fname msg)
                  ~hint:"the first argument must be `date(\"...\")`, \
                         `origin`, or a nested \
                         `add_calendar_months`/`add_calendar_years` call"
                  ();
                Ir.Const 0.0)))
     | _ ->
       Diagnostics.error ctx.diags
         ~code:"E328" ~loc:Diagnostics.no_loc
         ~message:(Printf.sprintf
           "%s expects two positional arguments: a date and an integer" fname)
         ~hint:(Printf.sprintf "example: %s(date(\"2020-01-01\"), 6)" fname)
         ();
       Ir.Const 0.0)
  | EFuncCall ("date_range", _) ->
    (* `date_range` produces an `Instant[]`. It is only legal in
       list-consuming positions (table values, intervention/event
       `at = [...]`, periodic-forcing `on = [...]`); those sites
       splice the expansion via [expand_date_range_to_consts]. A
       call that reaches the scalar `resolve_expr` path is in the
       wrong context. *)
    Diagnostics.error ctx.diags
      ~code:"E270" ~loc:Diagnostics.no_loc
      ~message:"`date_range(...)` produces a list and is only valid in \
                a list-consuming position"
      ~hint:"use date_range inside `at = [...]`, `on = [...]`, or a \
             table value — not in a scalar expression"
      ();
    Ir.Const 0.0
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
  | EFuncCall (("observed" | "sum_observed") as fname, _) ->
    (* gh#204: observed()/sum_observed() read policy-visible surveillance
       history and are valid ONLY inside a reactive trigger predicate (lowered
       by `lower_obs_quantity`, never through `resolve_expr`). Reaching here
       means one appeared in a rate / binding / other model expression — reject
       with a targeted message rather than the generic unknown-function error. *)
    Diagnostics.error ctx.diags ~code:"E278" ~loc:Diagnostics.no_loc
      ~message:(Printf.sprintf
        "'%s(...)' is only valid inside a reactive trigger predicate \
         (reactive_interventions { ... when %s(...) ... })" fname fname)
      ~hint:"trigger inputs read observed data, not latent model state — they \
             cannot appear in a transition rate or other model expression"
      ();
    Ir.Const 0.0
  | EFuncCall (name, _) when is_temporal_reduction_name name ->
    (* proposal 2026-06-25: temporal reductions fold a whole trajectory and are
       valid ONLY inside a `quantities { }` block (lowered by the quantity
       classifier, never through resolve_expr). Reaching here means one appeared
       in a rate / binding / other per-instant expression. (max/min are exempt —
       they are legitimate binary operators everywhere.) *)
    Diagnostics.error ctx.diags ~code:"E290" ~loc:Diagnostics.no_loc
      ~message:(Printf.sprintf
        "'%s(...)' is a temporal reduction, valid only inside a `quantities { }` block"
        name)
      ~hint:"a temporal reduction (final, max, time_of_max, integral, …) folds a \
             whole trajectory; it has no meaning in a rate or other per-instant \
             expression"
      ();
    Ir.Const 0.0
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
    else if fname = "min" || fname = "max" then begin
      (* Binary min/max -> Ir.BinOp Min/Max. The IR, Rust eval
         (propensity.rs), dimcheck (Add|Sub|Min|Max|Mod share the
         same-dimension rule) and autodiff (subgradient: differentiate
         the active branch) already support these; this only exposes the
         DSL surface. *)
      let op = if fname = "min" then Ir.Min else Ir.Max in
      match args with
      | [("", a); ("", b)] ->
        Ir.BinOp { op; left = resolve_expr ctx env a; right = resolve_expr ctx env b }
      | _ ->
        Diagnostics.error ctx.diags ~code:"E101" ~loc:Diagnostics.no_loc
          ~message:(Printf.sprintf "built-in function '%s' takes exactly two arguments" fname)
          ~hint:(Printf.sprintf "usage: %s(a, b)" fname) ();
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
  | EObsAccess (stream, l) ->
    (* proposal 2026-06-25 (v1.1): `observations.<stream>` reads the simulated
       observation series and is valid ONLY inside a `quantities { }` block (the
       quantity classifier lowers it to Ir.QSObservation, never through
       resolve_expr). Reaching here means one appeared in a rate / binding /
       other per-instant expression. *)
    Diagnostics.error ctx.diags ~code:"E290" ~loc:(diag_loc_of_ast_ctx ctx l)
      ~message:(Printf.sprintf
        "observations.%s is only valid in a quantities block" stream)
      ~hint:"the simulated observation series has no value in a rate or other \
             per-instant expression; reduce it inside `quantities { }`"
      ();
    Ir.Const 0.0
  | ERunMember r ->
    (* A run-rooted `<run>.<ns>.<member>` reference is a contrast operand, valid
       ONLY inside a `contrasts { }` body (where a dedicated resolver lowers it
       to Ir.CRunMember). Reaching resolve_expr means it appeared in a rate /
       binding / init / other per-instant expression. (`E293` covers the
       distinct case of one written inside a `quantities { }` recipe.) *)
    let ns = (match r.ns with NsQuantities -> "quantities" | NsObservations -> "observations") in
    Diagnostics.error ctx.diags ~code:"E292" ~loc:(diag_loc_of_ast_ctx ctx r.loc)
      ~message:(Printf.sprintf
        "`%s.%s.%s` is a contrast operand, only valid inside a `contrasts { }` block"
        r.run ns r.member)
      ~hint:"a run-rooted reference has no value in a rate or other per-instant \
             expression; use it as an operand in a `contrasts { }` entry"
      ();
    Ir.Const 0.0

and resolve_ident_name ctx name ~loc =
  (* Per-observation aux data column (§3): inside a likelihood, a declared
     value-column name (other than the scored outcome) is a reference to that
     observation's auxiliary data — e.g. the binomial denominator `n = tested`.
     It is resolved by name by the Rust binder, NOT by model name resolution.
     A column name that also collides with a compartment/parameter/let/forcing
     is a hard error naming both (§3.1) — never a silent re-bind. *)
  if List.mem name ctx.obs_aux_cols then begin
    if Hashtbl.mem ctx.expanded_comp_tbl name
       || Hashtbl.mem ctx.comp_tbl name
       || Hashtbl.mem ctx.scalar_param_tbl name
       || is_expanded_indexed_param_name ctx name
       || Hashtbl.mem ctx.let_tbl name
       || Hashtbl.mem ctx.func_tbl name
    then
      Diagnostics.error ctx.diags
        ~code:"E279"
        ~loc
        ~message:(Printf.sprintf
          "observation column '%s' collides with a model name (compartment / \
           parameter / let / forcing of the same name)" name)
        ~hint:"rename the data column (upstream) or the model declaration so \
               the likelihood reference is unambiguous"
        ();
    Ir.ObsColumnRef name
  end else
  (* Name resolution order follows spec §26.10: compartments → parameters →
     lets → forcings. `check_declaration_names` (gh#117) already makes a name
     live in at most one namespace — a cross-namespace collision is a hard
     E278 before we get here — so this order never changes the *result* on a
     valid model; it pins the implementation to the documented precedence so
     a future gap in the collision check can't silently re-introduce
     let-wins-over-param resolution. *)
  (* 1. Known expanded compartment? *)
  if Hashtbl.mem ctx.expanded_comp_tbl name then Ir.Pop name
  else if Hashtbl.mem ctx.comp_tbl name then begin
    let expansions = expand_compartment_name ctx name in
    if List.length expansions = 1 then Ir.Pop (List.hd expansions)
    else Ir.PopSum expansions
  end
  (* 2. Parameter (scalar or fully-expanded indexed)? *)
  else if Hashtbl.mem ctx.scalar_param_tbl name then
    Ir.Param name
  else if is_expanded_indexed_param_name ctx name then
    Ir.Param name
  (* 3. Let binding? Inline it — unless it's a typed const (emitted as Param). *)
  else match Hashtbl.find_opt ctx.let_tbl name with
  | Some lb ->
    if lb.lkind <> None && is_const_expr lb.lbody then
      (* Typed const let → treat as parameter (dimcheck will see param_kind) *)
      Ir.Param name
    else if let_is_hoistable ctx lb && not ctx.suppress_hoist then
      (* Fix B: a state-only scalar let (e.g. a mixed-compartment total) is
         extracted into a shared binding once instead of inlined per use. *)
      register_hoisted_binding ctx name
        (fun () -> normalize_expr (resolve_expr ctx [] lb.lbody))
    else
      normalize_expr (resolve_expr ctx [] lb.lbody)
  | None ->
  (* 4. Forcing / time function? *)
  if Hashtbl.mem ctx.func_tbl name then
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
  else if name = "origin" then
    (* `origin` as a referenceable identifier — Phase 2 of the
       2026-05-22 typed-time proposal §1.1. Resolves to the t=0
       point in anchored mode (origin IS the zero); errors in
       unanchored mode pointing the user at adding an `origin =
       date(...)` declaration. *)
    (match ctx.origin with
     | Some _ -> Ir.Const 0.0
     | None ->
       Diagnostics.error ctx.diags
         ~code:"E327"
         ~loc
         ~message:"`origin` is not defined — model is unanchored"
         ~hint:"add `origin = date(\"YYYY-MM-DD\")` at the top of the file, \
                or use an explicit numeric/duration value here"
         ();
       Ir.Const 0.0)
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

(** Structured (dimension, level) selector for one expanded leaf — the by-name
    routing key shared by stratified observations (§4.2) and generated
    quantities. Parallels [name_parts_from_bindings] but keeps the dimension
    name alongside the level. Only [IBind]/[IConsec] carry a dimension; [IComp]
    (iterate over compartments) contributes no stratum pair. An unstratified
    binding list yields []. *)
let stratum_of_bindings ibs env : (string * string) list =
  List.filter_map (fun ib ->
    match ib with
    | IBind (v, d) | IConsec (v, _, d) ->
      (match List.assoc_opt v env with
       | Some level -> Some (d, level)
       | None -> None)
    | IComp _ -> None
  ) ibs

(** Transition-name expander — the incidence analogue of
    [expand_compartment_name]. Given a base transition name, return all
    fully-expanded IR transition names that base produces, in emission
    order, or [None] if no declared transition has that base name.

    The naming logic mirrors [emit_one] in [expand_transitions_counted]
    exactly: one name per index combo that survives the `where` guard,
    suffixed by [name_parts_from_bindings], with a per-branch suffix for
    [DstBranch] destinations. Keeping these two in lockstep is what makes
    `incidence(<stratified family>)` expand to an Add-fold over names that
    actually exist post-expansion (spec §25.4). *)
let expand_transition_name ctx tname : string list option =
  match List.find_opt (fun tr -> tr.trname = tname) ctx.transitions with
  | None -> None
  | Some tr ->
    let combos = cartesian_product tr.trindices ctx in
    let names =
      List.concat_map (fun env ->
        let pass_guard = match tr.trguard with
          | None   -> true
          | Some g -> eval_guard ctx env g
        in
        if not pass_guard then []
        else begin
          let parts = name_parts_from_bindings tr.trindices env in
          let base_name =
            if parts = [] then tr.trname
            else tr.trname ^ "_" ^ String.concat "_" parts
          in
          match tr.trdst with
          | DstSum _ -> [base_name]
          | DstBranch branches ->
            List.map (fun ((dst_ref, _weight) : (Ast.stoich_ref * Ast.expr)) ->
              let (dst_base, _idx) = dst_ref in
              base_name ^ "_" ^ dst_base
            ) branches
        end
      ) combos
    in
    Some names

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
    | GTab (_, idxs, _, _) -> List.iter check_ident idxs
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
  let relop_str = function
    | RLt -> "<" | RLe -> "<=" | RGt -> ">" | RGe -> ">=" | REq -> "==" | RNe -> "!=" in
  let operand_str = function GoNum f -> Printf.sprintf "%g" f | GoName n -> n in
  let rec pp = function
    | GEq  (a, b) -> Printf.sprintf "%s == %s" a b
    | GNeq (a, b) -> Printf.sprintf "%s != %s" a b
    | GTab (t, idxs, op, v) ->
      Printf.sprintf "%s[%s] %s %s" t (String.concat "," idxs) (relop_str op) (operand_str v)
    | GAnd (g1, g2) -> Printf.sprintf "%s and %s" (pp g1) (pp g2)
    | GOr  (g1, g2) -> Printf.sprintf "%s or %s"  (pp g1) (pp g2)
  in
  pp g

(* Combine signed stoichiometry entries: sum each compartment's deltas, drop any
   that net to zero (a catalyst appearing on both sides), and preserve
   first-appearance order. Pulled out of [emit_one] so that function reads as a
   sequence of named steps rather than an inline Hashtbl fold. *)
let collapse_stoichiometry (entries : (string * int) list) : (string * int) list =
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

(* The single compartment carrying the requested sign in a collapsed
   stoichiometry (negative = source, positive = destination), or [None] when
   there are zero or several — the source/dest metadata is filled only for the
   unambiguous single-source / single-dest case. *)
let sole_with_sign (stoich : (string * int) list) (sign : int) : string option =
  let matches =
    List.filter (fun (_, d) -> if sign < 0 then d < 0 else d > 0) stoich in
  match matches with
  | [(n, _)] -> Some n
  | _        -> None

(* Lineage analysis (#[lineage], 2026-05-19 proposal). Runs on the
   resolved/normalized IR rate so Pop names are fully qualified after
   stratification, and the source set is the expanded source compartments. A
   nonlinear use of a parent count is rejected with E601 pointing at the
   transition; a structurally-valid (empty-weights) lineage record is still
   emitted so compilation continues collecting diagnostics, while the E601
   blocks a successful compile. *)
let resolve_lineage (ctx : context) (tr : Ast.transition_decl)
    ~(sources : string list) ~(tr_name : string) (rate : Ir.expr) : Ir.transition_lineage option =
  if not tr.trlineage then None
  else begin
    let cls = Lineage.classify_parents ~sources rate in
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
      Some { Ir.is_lineage_event = true; Ir.parent_pool_weights = [] }
    | None ->
      let weights = Lineage.parent_pool_weights ~sources rate in
      Some { Ir.is_lineage_event = true; Ir.parent_pool_weights = weights }
  end

let expand_transitions_counted ctx =
  let filtered = ref 0 in
  let expanded = List.concat_map (fun tr ->
    let combos = cartesian_product tr.trindices ctx in
    let tr_filtered = ref 0 in
    let results = List.map (fun env ->
      let pass_guard = match tr.trguard with
        | None   -> true
        | Some g -> eval_guard ctx env g
      in
      if not pass_guard then (incr filtered; incr tr_filtered; [])
      else begin
        let src_names = List.map (resolve_stoich_ref ctx env) tr.trsrc in
        (* By here a `via` transition has been desugared by [lower_via_transitions]
           (run before this pass), so the dynamics is always an ordinary `@ rate`.
           A surviving `Via` is a compiler invariant violation, not user error —
           the lowering should have rewritten or removed it. A located E243 (a
           hard error) still blocks the compile rather than emitting a silent
           rate-0 transition; reaching it means the pre-pass missed a case. *)
        let rate_expr =
          match tr.trdyn with
          | Rate e -> e
          | Via (law_name, _) ->
            Diagnostics.error ctx.diags
              ~code:"E001"
              ~loc:(diag_loc_of_ast_ctx ctx tr.trloc)
              ~message:(Printf.sprintf
                "internal error: transition '%s' still carries `via %s(...)` \
                 after staged-residence lowering" tr.trname law_name)
              ~hint:"this is a compiler bug — the `via` lowering pre-pass should \
                     have rewritten this transition; please report it"
              ();
            EConst 0.0
        in
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
        let raw_rate, draw_method = match rate_expr with
          | EFuncCall ("overdispersed", [("", inner); ("", var)]) ->
            let resolved_var = normalize_expr (resolve_expr ctx env var) in
            (inner, Ir.DrawOverdispersed { sigma_sq = resolved_var; sigma_sq_grad = [] })
          | EFuncCall ("overdispersed", args) ->
            validate_draw_shape "overdispersed" args 2
              "overdispersed(rate, sigma_squared)";
            (rate_expr, Ir.DrawPoisson)
          | EFuncCall ("deterministic", [("", inner)]) ->
            (inner, Ir.DrawDeterministic)
          | EFuncCall ("deterministic", args) ->
            validate_draw_shape "deterministic" args 1
              "deterministic(rate)";
            (rate_expr, Ir.DrawPoisson)
          | _ -> (rate_expr, Ir.DrawPoisson)
        in
        (* Build one IR transition given resolved destinations and a
           (possibly weight-scaled) raw rate. `name_suffix` gets
           appended to the transition name — used for branches to
           disambiguate `infect` → `infect_symp` / `infect_asym`. *)
        let emit_one dst_refs raw_rate_for_branch name_suffix =
          let dst_names = List.map (resolve_stoich_ref ctx env) dst_refs in
          (* Fix B: a #[lineage] rate must stay fully inlined so the parent
             decomposition can read its Pop structure (it cannot see through a
             BindingRef). Non-lineage transitions still extract the same let. *)
          let rate =
            let prev = ctx.suppress_hoist in
            ctx.suppress_hoist <- tr.trlineage;
            let r = normalize_expr (resolve_expr ctx env raw_rate_for_branch) in
            ctx.suppress_hoist <- prev;
            r
          in
          let raw_entries =
            List.map (fun n -> (n, -1)) src_names
            @ List.map (fun n -> (n,  1)) dst_names
          in
          let stoich = collapse_stoichiometry raw_entries in
          let src_meta = sole_with_sign stoich (-1) in
          let dst_meta = sole_with_sign stoich   1  in
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
          let lineage = resolve_lineage ctx tr ~sources:src_names ~tr_name rate in
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
            Ir.rate_state_grad = [];  (* populated later by the WrtPop pass (gh#275) *)
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

(* ── Parameter expansion ─────────────────────────────────────────────────── *)

(* resolve_float_expr_simple / resolve_bounds are defined below, after
   eval_const_expr, so bounds can const-evaluate negative/arithmetic literals
   (e.g. `in [-40, 120]` for an `instant` parameter). *)

let param_kind_to_string = function
  | PRate        -> "rate"
  | PProbability -> "probability"
  | PPositive    -> "positive"
  | PCount       -> "count"
  | PReal        -> "real"
  | PInstant     -> "instant"
  | PDuration    -> "duration"

(* AST parameter-kind → typed IR [Ir.param_kind] enum (the gh#191 type-swap).
   Distinct from [param_kind_to_string], which lowers to the free-string
   [table.cell_kind] field still kept as a string. *)
let ir_param_kind_of_ast : Ast.param_type -> Ir.param_kind = function
  | PRate        -> Ir.Rate
  | PProbability -> Ir.Probability
  | PPositive    -> Ir.Positive
  | PCount       -> Ir.Count
  | PReal        -> Ir.Real
  | PInstant     -> Ir.Instant
  | PDuration    -> Ir.Duration

(* Resolve a parameter's [Ir.param_dim] (the explicit (P,T) annotation) from
   the optional bracket annotation [pdim] and the optional tier-3 unit literal
   [punit] (gh#60). A unit literal is sugar for the dimension half of the
   bracket annotation: `positive 'ratio` ≡ `positive [1]`, `positive 'per_year`
   ≡ `positive [T^-1]`, `real 'count` ≡ `real [P]`. The scale half of the unit
   plays no role for a parameter — its value is always supplied in model time
   units (spec §2.4). The unit literal is only meaningful on the
   dimension-under-determined kinds (`positive`, `real`); on a kind whose
   dimension the keyword already fixes it is rejected (E281), and it may not be
   combined with a redundant/conflicting bracket annotation (E282). *)
let resolve_param_dim ctx ~loc ~pname (pkind : Ast.param_type)
    (pdim : (int * int) option) (punit : unit_lit option) : (int * int) option =
  match punit with
  | None -> pdim
  | Some u ->
    (match pkind with
     | PPositive | PReal -> ()
     | _ ->
       Diagnostics.error ctx.diags
         ~code:"E281"
         ~loc
         ~message:(Printf.sprintf
           "parameter '%s': a unit literal ('%s) is only allowed on the \
            'positive' and 'real' kinds"
           pname (unit_lit_to_string u))
         ~hint:(Printf.sprintf
           "the '%s' kind already fixes the dimension; drop the unit literal"
           (Ir.param_kind_name (ir_param_kind_of_ast pkind)))
         ());
    (match pdim with
     | None -> ()
     | Some _ ->
       Diagnostics.error ctx.diags
         ~code:"E282"
         ~loc
         ~message:(Printf.sprintf
           "parameter '%s': cannot give both a unit literal ('%s) and a \
            bracket dimension annotation"
           pname (unit_lit_to_string u))
         ~hint:"use one or the other — a unit literal already supplies the \
                dimension"
         ());
    Some (unit_lit_to_dim u)

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

(* Resolve a bound expression to a float. Bounds are compile-time constants —
   a numeric literal or arithmetic of literals, possibly **negative** (e.g. a
   seed time `tau : instant in [-40, 120]` that may fall before the origin).
   Const-evaluates via `eval_const_expr` so negated/arithmetic literals resolve
   correctly; falls back to 0.0 for a genuinely non-constant bound. *)
let resolve_float_expr_simple ctx e =
  if is_const_expr e then eval_const_expr ctx e
  else
    match normalize_expr (resolve_expr ctx [] e) with
    | Ir.Const f -> f
    | _ -> 0.0

let resolve_bounds ctx pbounds =
  match pbounds with
  | None -> None
  | Some (lo_e, hi_e) ->
    Some (resolve_float_expr_simple ctx lo_e, resolve_float_expr_simple ctx hi_e)

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
  | "log_uniform" -> Some ["lower"; "upper"]
  (* truncated_normal takes only mean/sd; its truncation bounds are read
     from the parameter's `in [lo, hi]` declaration (one source of truth). *)
  | "truncated_normal" -> Some ["mean"; "sd"]
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
  | "log_uniform" ->
    (match find "lower", find "upper" with
     | Some lo, _ when lo <= 0.0 ->
       Some (Printf.sprintf "log_uniform requires lower > 0 (got lower=%g); \
                             it is uniform on the log scale" lo)
     | _, Some hi when hi <= 0.0 ->
       Some (Printf.sprintf "log_uniform requires upper > 0 (got upper=%g); \
                             it is uniform on the log scale" hi)
     | Some lo, Some hi when lo >= hi ->
       Some (Printf.sprintf "log_uniform requires lower < upper (got lower=%g, upper=%g)" lo hi)
     | _ -> None)
  | "truncated_normal" -> pos_check "sd"
  | _ -> None

type prior_classification =
  [ `Plain        of Ir.prior_dist
  | `Hierarchical of Ir.hierarchical_prior ]

let resolve_prior_spec ?(loc = Diagnostics.no_loc) ?(bounds = None) ctx ~pname (ps : prior_spec) : Ir.prior_dist =
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
        ~detail:"Valid distributions: uniform, normal, log_normal, half_normal, beta, gamma, exponential, log_uniform, truncated_normal."
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
    | "log_uniform" -> Ir.LogUniform { Ir.lu_lower = v "lower"; Ir.lu_upper = v "upper" }
    | "truncated_normal" ->
      (* Truncation bounds come from the parameter's `in [lo, hi]`. *)
      (match bounds with
       | Some (lo, hi) ->
         Ir.TruncatedNormal { Ir.tn_mean = v "mean"; Ir.tn_sd = v "sd";
                              Ir.tn_lower = lo; Ir.tn_upper = hi }
       | None ->
         Diagnostics.error ctx.diags
           ~code:"E285" ~loc
           ~message:(qualify "truncated_normal requires explicit bounds")
           ~detail:"A truncated_normal prior is truncated to the parameter's \
                    declared range, but this parameter has no `in [lo, hi]` bounds."
           ~hint:"Add bounds, e.g. `take : probability in [0.3, 1.0] ~ truncated_normal(mean = 0.7, sd = 0.2)`."
           ();
         err_invalid_placeholder)
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
let classify_and_resolve_prior_spec ?(loc = Diagnostics.no_loc) ?(bounds = None) ctx ~pname
      (ps : prior_spec) : prior_classification =
  let has_non_const_arg =
    List.exists (fun (_, e) -> not (is_const_expr e)) ps.ps_args
  in
  let is_hierarchical = ps.ps_pool_over <> None || has_non_const_arg in
  if not is_hierarchical then
    `Plain (resolve_prior_spec ~loc ~bounds ctx ~pname ps)
  (* log_uniform and truncated_normal are constant-only distributions: they
     cannot reference hyperparameters or carry a `| dim` pooling clause.
     Without this guard they would reach [hierarchical_kind_of_name], which
     [failwith]s — an ICE instead of a diagnostic. *)
  else if ps.ps_name = "log_uniform" || ps.ps_name = "truncated_normal" then begin
    Diagnostics.error ctx.diags
      ~code:"E286" ~loc
      ~message:(Printf.sprintf
        "parameter '%s': prior '%s' cannot be hierarchical or pooled" pname ps.ps_name)
      ~detail:(Printf.sprintf
        "'%s' takes only constant arguments — it cannot reference \
         hyperparameters or use a `| dim` pooling clause." ps.ps_name)
      ~hint:"Use constant arguments, or choose a poolable distribution \
             (normal, log_normal, half_normal, beta, gamma, exponential)."
      ();
    `Plain (Ir.Uniform { Ir.lower = 0.0; Ir.upper = 1.0 })
  end
  else begin
    (* Validate distribution name but allow parameter references in args. *)
    let qualify msg = Printf.sprintf "parameter '%s': %s" pname msg in
    (match prior_arg_signature ps.ps_name with
     | Some _ -> ()
     | None ->
       Diagnostics.error ctx.diags
         ~code:"E232" ~loc
         ~message:(qualify (Printf.sprintf "unknown prior distribution '%s'" ps.ps_name))
         ~detail:"Valid distributions: uniform, normal, log_normal, half_normal, beta, gamma, exponential, log_uniform, truncated_normal."
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
      | Ir.Param _ | Ir.Const _ | Ir.Projected | Ir.ObsColumnRef _ | Ir.Time | Ir.Dt -> ()
      | Ir.BinOp b -> check_refs b.left; check_refs b.right
      | Ir.UnOp  u -> check_refs u.arg
      | Ir.Cond  c -> check_refs c.pred; check_refs c.then_; check_refs c.else_
      | Ir.Pop _ | Ir.PopSum _ -> ()  (* caught elsewhere *)
      | Ir.TimeFunc _ -> ()
      | Ir.TableLookup (_, args) -> List.iter check_refs args
      | Ir.UncheckedDim u -> check_refs u.inner
      | Ir.Reduce terms -> List.iter check_refs terms
      | Ir.BindingRef _ -> ()
      | Ir.PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"
    in
    List.iter (fun (_, e) -> check_refs e) resolved_args;
    `Hierarchical {
      Ir.hkind      = Ir.hierarchical_kind_of_name ps.ps_name;
      Ir.hargs      = resolved_args;
      Ir.hpool_over = Option.value ~default:"" ps.ps_pool_over;
    }
  end

(* Map a `parameters {}` declaration (which never carries a `= value`, so
   the IR value is never Fixed here) to the typed [Ir.param_value]:
   - any inference config (bounds and/or a prior) → [Estimated] carrying it;
   - a bare `name : kind` with neither → [Required] (supplied at runtime).
   The compiler emits transform [Identity] and no init; the fit layer derives
   the real transform/start per-fit. (Typed-const `let`s are [Fixed], below.) *)
let mk_estimated_or_required ~bounds ~prior ~hierarchical : Ir.param_value =
  match bounds, prior, hierarchical with
  | None, None, None -> Ir.Required
  | _ ->
    let est_prior = match prior, hierarchical with
      | Some p, _      -> Ir.Dist p
      | _, Some h      -> Ir.Hierarchical h
      | None, None     -> Ir.Flat
    in
    Ir.Estimated { est_init = None; est_bounds = bounds; est_prior;
                   est_transform = Ir.Identity }

(* Convert an AST `#'` doc block into its IR mirror (presentation metadata). *)
let ir_doc_of_ast (d : Ast.doc option) : Ir.doc option =
  Option.map (fun (a : Ast.doc) ->
    { Ir.text = a.Ast.d_text; symbol = a.d_symbol; reference = a.d_ref }) d

(* Fold every source declaration's `#'` doc into the model's doc dictionary,
   keyed by base declaration name (matching what an author wrote and the logical
   names ObsSchema groups by). Only documented declarations appear. *)
let build_doc_index ctx : Ir.doc_index =
  let collect get_name get_doc decls =
    List.filter_map (fun d ->
      match ir_doc_of_ast (get_doc d) with
      | Some doc -> Some (get_name d, doc)
      | None     -> None) decls
  in
  { Ir.di_parameters =
      collect (function Ast.PScalar s -> s.pname | Ast.PIndexed i -> i.pname)
              (function Ast.PScalar s -> s.pdoc  | Ast.PIndexed i -> i.pdoc)
              ctx.param_decls;
    Ir.di_compartments =
      collect (fun (c : Ast.compartment_decl) -> c.cname) (fun c -> c.cdoc) ctx.comp_decls;
    Ir.di_transitions =
      collect (fun (t : Ast.transition_decl) -> t.trname) (fun t -> t.trdoc) ctx.orig_transitions;
    Ir.di_observations =
      collect (fun (o : Ast.obs_decl) -> o.oname) (fun o -> o.odoc) ctx.obs_decls;
    Ir.di_dimensions =
      collect (fun (d : Ast.dimensions_entry) -> d.dename) (fun d -> d.dedoc) ctx.dim_decls;
    Ir.di_quantities =
      collect (fun (q : Ast.quantity_decl) -> q.qd_name) (fun q -> q.qd_doc) ctx.quantity_decls;
  }

let expand_parameters ctx =
  let from_params = List.concat_map (fun pd ->
    match pd with
    | PScalar { pname; pbounds; pkind; pdim; punit; pprior; ploc; _ } ->
      let bounds = resolve_bounds ctx pbounds in
      let pk = Some (ir_param_kind_of_ast pkind) in
      let loc = diag_loc_of_ast_ctx ctx ploc in
      let dim = resolve_param_dim ctx ~loc ~pname pkind pdim punit in
      let (prior, hierarchical) = match pprior with
        | None -> (None, None)
        | Some ps -> (match classify_and_resolve_prior_spec ctx ~loc ~bounds ~pname ps with
                      | `Plain p        -> (Some p, None)
                      | `Hierarchical h -> (None, Some h))
      in
      [{ Ir.name       = pname;
         Ir.value      = mk_estimated_or_required ~bounds ~prior ~hierarchical;
         Ir.param_kind = pk;
         Ir.param_dim  = dim;
       }]
    | PIndexed { pname; pdims = [dim]; pbounds; pkind; pdim = pdim_ann; punit; pprior; ploc; _ } ->
      let vals = dim_values ctx dim in
      let bounds = resolve_bounds ctx pbounds in
      let pk = Some (ir_param_kind_of_ast pkind) in
      let loc = diag_loc_of_ast_ctx ctx ploc in
      let resolved_dim = resolve_param_dim ctx ~loc ~pname pkind pdim_ann punit in
      let (prior, hierarchical) = match pprior with
        | None -> (None, None)
        | Some ps -> (match classify_and_resolve_prior_spec ctx ~loc ~bounds ~pname ps with
                      | `Plain p        -> (Some p, None)
                      | `Hierarchical h -> (None, Some h))
      in
      List.map (fun v ->
        { Ir.name       = pname ^ "_" ^ v;
          Ir.value      = mk_estimated_or_required ~bounds ~prior ~hierarchical;
          Ir.param_kind = pk;
          Ir.param_dim  = resolved_dim;
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
      Some { Ir.name       = lb.lname;
             Ir.value      = Ir.Fixed v;
             Ir.param_kind = Some (ir_param_kind_of_ast pk);
             Ir.param_dim  = None;
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
  | EList es     ->
    (* Splice date_range(...) calls inside list literals before
       recursing — Phase 2 of the 2026-05-22 typed-time proposal §4. *)
    let es = splice_date_ranges ctx es in
    List.concat_map (flatten_expr_list ctx dim_entries) es
  | EFuncCall ("date_range", args) ->
    (* date_range at the top of a table value (without an outer
       []) also splices. *)
    List.concat_map (flatten_expr_list ctx dim_entries)
      (expand_date_range_to_consts ctx args)
  | EConst f     -> [Ir.Const f]
  | EUnit (f, u) -> [Ir.Const (unit_to_model_time ctx f u)]
  | other        -> [resolve_expr ctx [] other]

(** Determine table source: External if `external("name")`, otherwise Inline. *)
let table_source_of_expr ?table_name ctx (dim_entries : table_dim_entry list) e =
  match e with
  | EFuncCall ("external", args) ->
    (match extract_path_arg ctx "external" args with
     | None -> Ir.Inline []
     | Some name -> Ir.External name)
  | _ ->
    let vals = flatten_expr_list ctx dim_entries e in
    (* gh#112: an inline table flattens to a row-major value list. Verify the
       cell count equals the product of declared dimension sizes — a too-short
       table would otherwise fail late (or read a default) and a too-long one
       would silently carry unused data. Only checked when every declared dim
       has known levels (a dim error is reported elsewhere). *)
    let dim_sizes = List.map (fun de ->
      List.length (dim_values ctx (dim_name_of_entry de))) dim_entries in
    let expected = List.fold_left ( * ) 1 dim_sizes in
    (if dim_entries <> [] && not (List.exists (fun s -> s = 0) dim_sizes)
        && List.length vals <> expected then
       Diagnostics.error ctx.diags
         ~code:"E202" ~loc:Diagnostics.no_loc
         ~message:(Printf.sprintf
           "table '%s' declares dimensions [%s] (%d cells) but its inline \
            value has %d cell%s"
           (match table_name with Some n -> n | None -> "<table>")
           (String.concat " \xc3\x97 " (List.map dim_name_of_entry dim_entries))
           expected (List.length vals)
           (if List.length vals = 1 then "" else "s"))
         ~hint:"an inline table must list exactly one value per cell, in \
                row-major order (last dimension varies fastest)"
         ());
    Ir.Inline vals

let expand_tables ctx =
  List.concat_map (fun td ->
    let dim_entries = td.tdims in
    let primary_name = List.hd td.tnames in
    let table_unit = extract_table_unit ctx ~table_name:primary_name dim_entries in
    let cell_kind = Option.map param_kind_to_string td.tcell_kind in
    match td.tvalue with
    | EFuncCall ("read", _) when dim_entries = [] ->
      (* A `read(...)` with no index dimensions is the scalar-via-table
         mistake: tables hold indexed data and require >=1 dimension. The
         home for an externally-computed scalar is a parameter (params.toml /
         --params, which a preprocessing pipeline can generate), not a table
         read. Diagnose the seam here, before the loader trips a confusing
         file/column error (E200/E206). *)
      Diagnostics.error ctx.diags
        ~code:"E222" ~loc:Diagnostics.no_loc
        ~message:(Printf.sprintf
          "table '%s' uses read(...) but declares no index dimensions; read() loads \
           indexed tables (one column per dimension) and needs at least one dimension. \
           For a single externally-computed scalar, declare '%s' as a parameter, not a table."
          primary_name primary_name)
        ~hint:(Printf.sprintf
          "supply the value via params.toml or `--params %s=<value>` (a value a \
           preprocessing pipeline can generate); read() loads indexed tables, not scalars"
          primary_name)
        ();
      []
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
           ~dims:dim_entries ~n_values ~default_val ~cell_kind in
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
      let source = table_source_of_expr ~table_name:name ctx dim_entries td.tvalue in
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

(* Index the resolved constant tables by name → (row-major flattened cells,
   ordered dimension names), so a compile-time `where` predicate can read a
   cell's value during [resolve_expr]. Built once, right after [expand_tables]
   and before transition expansion. External (`--table`) tables have no
   compile-time values and are skipped (a predicate over one then errors with
   E284 — "not a compile-time-constant table"). *)
let build_table_index ctx (tables : Ir.table list) : unit =
  Hashtbl.reset ctx.table_index;
  let dim_name = function TDim s | TDimUnit (s, _) -> s in
  List.iter (fun (t : Ir.table) ->
    match t.Ir.source with
    | Ir.Inline cells ->
      let dims =
        match List.find_opt (fun (td : table_decl) -> List.mem t.Ir.name td.tnames)
                ctx.table_decls with
        | Some td -> List.map dim_name td.tdims
        | None    -> []
      in
      Hashtbl.replace ctx.table_index t.Ir.name (Array.of_list cells, dims)
    | Ir.External _ -> ()
  ) tables

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
  (* gh#114: every emitted init key must name a real expanded compartment.
     `expand_init` previously emitted a bare/concatenated key with no check,
     so `init { S = N0 }` against a stratified `S` produced an init entry
     named `S` (which is not an expanded cell — the real cells are
     `S_child`, `S_adult`), silently starting those cells at 0. Distinguish
     the two failure shapes for a precise, located E277. We emit exactly ONE
     diagnostic per offending entry (reviewer feedback: the prior attempt
     emitted a no-location E513 then a located E277 for one root cause). *)
  let check_membership ie concrete_name =
    if not (Hashtbl.mem ctx.expanded_comp_tbl concrete_name) then begin
      let loc = diag_loc_of_ast_ctx ctx ie.iloc in
      if Hashtbl.mem ctx.comp_tbl ie.icomp then begin
        (* The base compartment exists, but this key is not a real cell:
           a bare stratified reference, or a wrong/partial index set. *)
        let cells = expand_compartment_name ctx ie.icomp in
        Diagnostics.error ctx.diags ~code:"E277" ~loc
          ~message:(Printf.sprintf
            "initial condition '%s' does not name an expanded compartment \
             cell — compartment '%s' is stratified and requires explicit \
             strata in `init`"
            concrete_name ie.icomp)
          ~hint:(Printf.sprintf
            "specify each cell, e.g. %s = ...   (cells: %s)"
            (List.hd cells) (String.concat ", " cells))
          ()
      end else
        Diagnostics.error ctx.diags ~code:"E277" ~loc
          ~message:(Printf.sprintf
            "initial condition '%s' names an unknown compartment" concrete_name)
          ~hint:"check the `compartments` block; init keys must be real \
                 (expanded) compartment cells"
          ()
    end
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
      check_membership ie concrete_name;
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
        check_membership ie concrete_name;
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
      Ir.time_semantics = "continuous"; Ir.dt = None; Ir.rng_seed = None;
      Ir.integrator = Ir.Rk4 }
  | Some sd ->
    let t_start = resolve_float_expr ctx sd.sim_from in
    let t_end   = resolve_float_expr ctx sd.sim_to   in
    (* gh#161: `dt` is a model knob. It is unit-aware like from/to —
       `dt = 0.05 'months` resolves through resolve_float_expr (EUnit →
       model time units). None when omitted, so the CLI default / --dt
       override applies. *)
    let dt = Option.map (resolve_float_expr ctx) sd.sim_dt in
    (* gh#166: build the tagged integrator. atol/rtol are DIMENSIONLESS adaptive
       tolerances (ratios, not times). dimcheck does not visit the simulate
       block, so the dimension is checked here by computing the expression's NET
       (population, time) dimension — NOT by AST shape: a bare `1e-8 'days`, a
       composed `(1e-8) 'days`, and `0.5 * 1 'day` are all rejected, while a
       dimensionless `'ratio` unit is accepted. rk4 takes NO tolerances (the
       grammar permits `rk4 { ... }`; reject it semantically). *)
    let rec tol_dim e : (int * int) option =
      (* None when a sub-term is not a unit-bearing constant (e.g. a named
         binding); those fall through to resolve_float_expr unchanged. *)
      match e with
      | EConst _       -> Some (0, 0)
      | EUnit (_, u)   -> Some (unit_lit_to_dim u)
      | EUnOp (Neg, a) -> tol_dim a
      | EBinOp ((Add | Sub), a, b) ->
        (match tol_dim a, tol_dim b with
         | Some da, Some db when da = db -> Some da
         | _ -> None)
      | EBinOp (Mul, a, b) ->
        (match tol_dim a, tol_dim b with
         | Some (pa, ta), Some (pb, tb) -> Some (pa + pb, ta + tb)
         | _ -> None)
      | EBinOp (Div, a, b) ->
        (match tol_dim a, tol_dim b with
         | Some (pa, ta), Some (pb, tb) -> Some (pa - pb, ta - tb)
         | _ -> None)
      | _ -> None
    in
    let resolve_tol name = function
      | None -> None
      | Some (e, eloc) ->
        (match tol_dim e with
         | Some d when d <> (0, 0) ->
           Diagnostics.error ctx.diags ~code:"E106"
             ~loc:(diag_loc_of_ast_ctx ctx eloc)
             ~message:(Printf.sprintf
               "`%s` must be dimensionless: drop the unit (it is a tolerance, not a time)" name)
             ~hint:(Printf.sprintf "write `%s = 1e-8`" name) ();
           None
         | _ -> Some (resolve_float_expr ctx e))
    in
    let integrator =
      match sd.sim_integrator with
      | None -> Ir.Rk4   (* no integrator key: tolerances cannot be parsed without one *)
      | Some ("rk4", mloc) ->
        if sd.sim_atol <> None || sd.sim_rtol <> None then
          Diagnostics.error ctx.diags ~code:"E106"
            ~loc:(diag_loc_of_ast_ctx ctx mloc)
            ~message:"`integrator = rk4` takes no tolerances (atol/rtol are rk45-only)"
            ~hint:"write `integrator = rk45 { atol = .., rtol = .. }`" ();
        Ir.Rk4
      | Some ("rk45", _) ->
        Ir.Rk45 { atol = resolve_tol "atol" sd.sim_atol;
                  rtol = resolve_tol "rtol" sd.sim_rtol }
      | Some (other, mloc) ->
        Diagnostics.error ctx.diags ~code:"E106"
          ~loc:(diag_loc_of_ast_ctx ctx mloc)
          ~message:(Printf.sprintf "unknown integrator '%s': expected `rk4` or `rk45`" other)
          ~hint:"`integrator = rk4` or `integrator = rk45 { atol = .., rtol = .. }`" ();
        Ir.Rk4
    in
    { Ir.t_start; Ir.t_end;
      Ir.time_semantics = "continuous"; Ir.dt; Ir.rng_seed = None;
      Ir.integrator }

let expand_output ctx =
  (* The output window's upper bound is no longer stored on the schedule
     (gh#143): `simulation.t_end` is the sole horizon authority, and the
     runtime derives output times from `[start, t_end]` at emission. Only
     `start` is set here — a deliberate widening to `min(0, t_start)`. *)
  let t_start = match ctx.simulate with
    | None    -> 0.0
    | Some sd -> resolve_float_expr ctx sd.sim_from
  in
  let format = match ctx.output_decl with
    | Some { out_trajectories = Some ot; _ } -> ot.otformat
    | _ -> "tsv"
  in
  (* Default the output schedule's start to cover the full integration
     window. With anchored models that use `from = date(...)` before
     `origin`, t_start is negative; an output schedule starting at 0
     leaves no snapshots in [t_start, 0), and `--obs-only` / any
     state-at-obs-time path (snap_at) can't find a snapshot for the
     pre-origin observations and hard-exits. Defaulting to
     min(0.0, t_start) preserves the existing start=0 behaviour for
     unanchored models (t_start ≥ 0) and extends it to cover negative
     t_start without changing the step or output cadence. (start applies
     only to the regular schedule; `at = [...]` lists explicit times.) *)
  let start = min 0.0 t_start in
  let times = match ctx.output_decl with
    | Some { out_trajectories = Some ot; _ } ->
      (match ot.otschedule with
       | SchedEvery e -> Ir.OutRegular { Ir.start; Ir.step = resolve_float_expr ctx e }
       | SchedAt ts   -> Ir.OutAtTimes (List.map (resolve_float_expr ctx) ts))
    | _ -> Ir.OutRegular { Ir.start; Ir.step = 1.0 }
  in
  { Ir.times;
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
      | Ir.ObsColumnRef c -> Printf.sprintf "an observation data column ('%s')" c
      | Ir.Pop _      -> "a compartment" (* unreachable by pattern *)
      | Ir.UncheckedDim _ -> "a dimensional-escape expression"
      | Ir.Reduce _   -> "a sum (reduce)"
      | Ir.BindingRef _ -> "a binding reference"
      | Ir.PerEvalRef _ -> "a per-eval binding reference"
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
  (* A forcing's time_col is internal time. A bare number is taken directly; an
     ISO date (YYYY-MM-DD) resolves via the model's origin + time_unit — the
     same rule observation-data time columns and instant/duration tables follow
     (docs/dates.md, "The one rule"). A failure is a located E209, never a
     silent drop: dropping the time while keeping the value desynced the two
     arrays and the runtime then interpolated to 0 everywhere (gh#308). *)
  let parse_time_cell row_num cell =
    match float_of_string_opt cell with
    | Some t -> Some t
    | None ->
      (match parse_iso_date cell, ctx.origin with
       | Ok _, Some origin_str ->
         (try Some (parse_date_to_float origin_str cell ctx.time_unit)
          with Failure msg | Invalid_argument msg ->
            Diagnostics.error ctx.diags ~code:"E209" ~loc:Diagnostics.no_loc
              ~message:(Printf.sprintf
                "%s row %d: time_col cell '%s' is not a number or resolvable ISO date (%s)"
                path row_num cell msg)
              ();
            None)
       | Ok _, None ->
         Diagnostics.error ctx.diags ~code:"E209" ~loc:Diagnostics.no_loc
           ~message:(Printf.sprintf
             "%s row %d: time_col has date cell '%s' but the model declares no origin"
             path row_num cell)
           ~hint:"add a top-level `origin = date(\"YYYY-MM-DD\")`, or use numeric \
                  day-offsets in the time column"
           ();
         None
       | Error why, _ ->
         Diagnostics.error ctx.diags ~code:"E209" ~loc:Diagnostics.no_loc
           ~message:(Printf.sprintf
             "%s row %d: time_col cell '%s' is neither a number nor an ISO date (%s)"
             path row_num cell why)
           ();
         None)
  in
  let parse_value_cell row_num cell =
    match float_of_string_opt cell with
    | Some v -> Some v
    | None ->
      Diagnostics.error ctx.diags ~code:"E209" ~loc:Diagnostics.no_loc
        ~message:(Printf.sprintf
          "%s row %d: value_col cell '%s' is not a number" path row_num cell)
        ();
      None
  in
  let on_row row_num cols =
    let get i = String.trim (try List.nth cols i with _ -> "") in
    if !key_ci < 0 || get !key_ci = key_val then
      (* Push time and value together so a parse failure cannot desync the two
         arrays — a diagnostic has already been emitted, which fails the compile. *)
      match parse_time_cell row_num (get !time_ci),
            parse_value_cell row_num (get !value_ci) with
      | Some t, Some v ->
        times  := t :: !times;
        values := v :: !values
      | _ -> ()
  in
  let on_done () = (List.rev !times, List.rev !values) in
  match read_csv_rows ctx path
          ~ref_desc:"the forcing data reference"
          ~ref_hint_example:"data = \"data/forcing.tsv\""
          ~on_header ~on_row ~on_done with
  | Some result -> result
  | None -> ([], [])

let expand_time_function_one ctx fname (env : (string * string) list)
    (findices : index_binding list) fkind (funit : unit_lit) fargs =
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
  (* gh#345: a table-backed interpolated forcing draws its per-stratum series
     from a `tables {}` matrix. `table = temp_data` names the source; `time_dim =
     week` names which dimension is the time axis; the forcing's own index binds
     the stratum. Every table dimension must be either indexed by the forcing or
     the `time_dim`, else a named error. Returns (time, value-cell) knots sorted
     by time — a dimension's level order is arbitrary, and the Rust knot-builder
     requires a strictly-increasing axis (and rejects duplicate times). *)
  let table_backed_knots tbl =
    let err ?hint msg =
      Diagnostics.error ctx.diags ~code:"E229" ~loc:Diagnostics.no_loc ~message:msg ?hint () in
    let time_dim = get_str_kw "time_dim" "" in
    let tdims    = table_dims ctx tbl in
    let stratum  = List.filter_map (function IBind (v, d) -> Some (v, d) | _ -> None) findices in
    if tdims = [] then
      (err ~hint:(Printf.sprintf "declare it as `tables { %s : dim \xc3\x97 time = read(...) }`" tbl)
         (Printf.sprintf "forcing '%s': `table = %s` is not a table" fname tbl); [])
    else if time_dim = "" then
      (err ~hint:"name the table dimension that is the time axis, e.g. `time_dim = week`"
         (Printf.sprintf "forcing '%s': a table-backed forcing needs `time_dim = <dimension>`" fname);
       [])
    else if List.length stratum <> List.length findices then
      (err (Printf.sprintf
         "forcing '%s': a table-backed forcing must be indexed by plain dimensions \
          (`[p in patch]`), not consecutive pairs or compartments" fname); [])
    else begin
      let stratum_dims = List.map snd stratum in
      (* Dimension accounting: { stratum dims } ∪ { time_dim } == { table dims }. *)
      if not (List.mem time_dim tdims) then
        err (Printf.sprintf
          "forcing '%s': time_dim = '%s' is not a dimension of table '%s' [%s]"
          fname time_dim tbl (String.concat " \xc3\x97 " tdims));
      List.iter (fun d ->
        if d <> time_dim && not (List.mem d stratum_dims) then
          err ~hint:(Printf.sprintf
            "index it (e.g. add `%s in %s` to the forcing) or aggregate '%s' out of the table"
            (String.lowercase_ascii d) d d)
            (Printf.sprintf
              "forcing '%s': table '%s' has dimension '%s' that is neither indexed by \
               the forcing nor the time axis" fname tbl d)) tdims;
      List.iter (fun d ->
        if not (List.mem d tdims) then
          err (Printf.sprintf
            "forcing '%s': indexed by dimension '%s', but table '%s' has no such dimension"
            fname d tbl)) stratum_dims;
      let cells = match Hashtbl.find_opt ctx.table_index tbl with
        | Some (arr, _) -> arr | None -> [||] in
      if Array.length cells = 0 then
        (err (Printf.sprintf
           "forcing '%s': table '%s' has no compile-time values to slice \
            (an external `--table` cannot be sliced)" fname tbl); [])
      else begin
        let sizes = List.map (fun d -> List.length (dim_values ctx d)) tdims in
        let stratum_pos d =
          match List.find_opt (fun (_, dd) -> dd = d) stratum with
          | Some (v, _) ->
            let lvl = match List.assoc_opt v env with Some l -> l | None -> v in
            int_of_float (dim_value_index ctx d lvl)
          | None -> 0
        in
        let pairs = List.mapi (fun j lvl ->
          let positions = List.map (fun d -> if d = time_dim then j else stratum_pos d) tdims in
          let off = row_major_offset positions sizes in
          let cell = if off >= 0 && off < Array.length cells then cells.(off) else Ir.Const 0.0 in
          let t = match float_of_string_opt lvl with
            | Some f -> f
            | None ->
              err ~hint:"the time axis must be a dimension whose levels are numbers"
                (Printf.sprintf
                  "forcing '%s': time_dim '%s' level '%s' is not numeric, so it cannot \
                   be a knot time" fname time_dim lvl);
              0.0
          in
          (t, cell)
        ) (dim_values ctx time_dim) in
        List.stable_sort (fun (a, _) (b, _) -> Float.compare a b) pairs
      end
    end
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
         (* The stratum filter. For an indexed forcing the level comes from its
            single index binding (env = [(binder_var, level)]); it is
            independent of both the binder name (`p`) and the data column name
            (`key_col`). A non-indexed forcing (env = []) reads every row. A
            forcing indexed by more than one dimension cannot be filtered by a
            single key column. *)
         let (key_col, key_val) = match env with
           | []                -> ("", "")
           | [ (_var, level) ] -> (get_str_kw "key_col" "key", level)
           | _ ->
             Diagnostics.error ctx.diags ~code:"E226" ~loc:Diagnostics.no_loc
               ~message:(Printf.sprintf
                 "file-backed forcing '%s' is indexed by %d dimensions, but a \
                  data file can only be filtered by a single key column"
                 fname (List.length env))
               ~hint:"index the forcing by one dimension, or pre-join the data \
                      to a single key column"
               ();
             (get_str_kw "key_col" "key", "")
         in
         let (times, values) =
           load_interpolated_for_level ctx path ~key_col ~key_val ~time_col ~value_col
         in
         (* An interpolated forcing with no knots silently interpolates to 0
            everywhere (gh#308). For an indexed forcing that almost always means
            the key filter matched nothing — fail here, naming the stratum, so
            the user is not left with the runtime's opaque "no knots" error. *)
         (if times = [] then
            if env = [] then
              Diagnostics.error ctx.diags ~code:"E227" ~loc:Diagnostics.no_loc
                ~message:(Printf.sprintf
                  "file-backed forcing '%s': '%s' contains no data rows" fname path)
                ~hint:"the file must have at least one (time, value) row"
                ()
            else
              Diagnostics.error ctx.diags ~code:"E227" ~loc:Diagnostics.no_loc
                ~message:(Printf.sprintf
                  "file-backed forcing '%s': no rows in '%s' where column '%s' = '%s'"
                  fname path key_col key_val)
                ~hint:"check that key_col names the column holding the stratum \
                       id and that the file has rows for every level"
                ());
         Ir.Interpolated {
           times   = List.map (fun f -> Ir.Const f) times;
           values  = List.map (fun f -> Ir.Const f) values;
           method_;
         }
       | _ ->
         match List.assoc_opt "table" fargs with
         | Some (EIdent (tbl, _)) ->
           let knots = table_backed_knots tbl in
           Ir.Interpolated {
             times  = List.map (fun (t, _) -> Ir.Const t) knots;
             values = List.map (fun (_, c) -> c) knots;
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
          (* Rule 4 of the 2026-05-22 typed-time proposal: in
             anchored mode, bare-numeric entries inside `on=[...]`
             are a hard error. The legitimate intent — calendar-
             aligned breakpoints — wants date(...) entries (or
             unit-annotated offsets); bare numbers under
             origin = date(...) are almost never what the user
             means. Corpus survey shows zero anchored models use
             `on=[...]` today, so this breaks nothing. *)
          let anchored = ctx.origin <> None in
          let bare_numeric_on_endpoint (e : expr) : bool =
            match e with
            | EConst _ -> true
            | EUnOp (Neg, EConst _) -> true
            | _ -> false
          in
          List.iter (fun range ->
            match range with
            | ERange (lo_e, hi_e) ->
              if anchored && (bare_numeric_on_endpoint lo_e || bare_numeric_on_endpoint hi_e) then
                Diagnostics.error ctx.diags ~code:"E323" ~loc:Diagnostics.no_loc
                  ~message:(Printf.sprintf
                    "periodic forcing '%s': bare-numeric entries in `on=[...]` \
                     are not allowed under `origin = date(...)`"
                    fname)
                  ~hint:Time_typing.hint_bare_numeric_on_periodic
                  ();
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
  (* gh#314: optional `lag = <duration>` — an evaluation-time shift applied
     uniformly to every forcing kind (the runtime evaluates the forcing at
     `t − lag`). The kwarg value is resolved like any other forcing argument,
     so a unit-annotated literal (`10 'days`) is rescaled into the model's
     `time_unit` by `resolve_expr` (via `EUnit`/`unit_to_model_time`), and a
     bare parameter reference (`lag = tau`) is preserved as `Param`. Absent
     ⇒ `None` ⇒ no shift. The dim-checker validates that `lag` carries a time
     dimension. *)
  let lag = match List.assoc_opt "lag" fargs with
    | None   -> None
    | Some e -> Some (resolve_expr ctx env e)
  in
  (* Return the load-time rescale factor [scale] alongside the expanded forcing:
     it is the single source of the per-forcing scale (L403 gh#13 needs it to
     tell an actually-rescaled forcing from a same-unit one, and cannot recover
     it from [Ir.time_function], which retains only [dim]). *)
  ({ Ir.name = fname; Ir.kind; Ir.dim; Ir.lag }, scale)

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

(* Returns the expanded forcings and, alongside, an assoc list mapping each
   expanded forcing name to its load-time rescale factor (from
   [expand_time_function_one]). The scale map is consumed by [lint_l403]. *)
let expand_time_functions ctx : Ir.time_function list * (string * float) list =
  let pairs =
    List.concat_map (fun (fd : func_decl) ->
      if fd.findices = [] then
        [expand_time_function_one ctx fd.fname [] fd.findices fd.fkind fd.funit fd.fargs]
      else begin
        let combos = cartesian_product fd.findices ctx in
        List.map (fun env ->
          let parts = name_parts_from_bindings fd.findices env in
          let fname = fd.fname ^ "_" ^ String.concat "_" parts in
          expand_time_function_one ctx fname env fd.findices fd.fkind fd.funit fd.fargs
        ) combos
      end
    ) ctx.func_decls
  in
  (List.map fst pairs,
   List.map (fun ((tf : Ir.time_function), s) -> (tf.Ir.name, s)) pairs)

(* gh#204: the shared action resolver — used by both scheduled
   interventions/events and reactive policies, so the transfer-kwarg validation
   (E261/E262) and the set target check (E265) live in ONE place rather than
   forking across the two fire sources. [name] is the expanded instance name
   (for diagnostics); [loc] its source location. *)
let resolve_intervention_action ctx env ~name ~loc (action : action_decl) : Ir.action list =
  match action with
  | ATransfer kwargs ->
    let has_from     = List.mem_assoc "from"     kwargs in
    let has_to       = List.mem_assoc "to"       kwargs in
    let has_fraction = List.mem_assoc "fraction" kwargs in
    let has_count    = List.mem_assoc "count"    kwargs in
    let known = ["from"; "to"; "fraction"; "count"] in
    let unknown = List.filter_map (fun (k, _) ->
      if k = "" || List.mem k known then None else Some k) kwargs in
    let err code msg hint =
      Diagnostics.error ctx.diags ~code ~loc:Diagnostics.no_loc ~message:msg ~hint ()
    in
    if not has_from then
      err "E261" (Printf.sprintf
        "intervention '%s': transfer action missing `from =`" name)
        "example: transfer(from = S, to = V, fraction = 0.8)";
    if not has_to then
      err "E261" (Printf.sprintf
        "intervention '%s': transfer action missing `to =`" name)
        "example: transfer(from = S, to = V, fraction = 0.8)";
    if not (has_fraction || has_count) then
      err "E261" (Printf.sprintf
        "intervention '%s': transfer action needs either `fraction =` or \
         `count =`" name)
        "fraction = 0.0..1.0 (relative) OR count = N (absolute)";
    if has_fraction && has_count then
      err "E261" (Printf.sprintf
        "intervention '%s': transfer action has both `fraction` and `count` \
         — these are mutually exclusive" name)
        "pick one: fraction for a proportion, count for an absolute number";
    List.iter (fun k ->
      err "E262" (Printf.sprintf
        "intervention '%s': unknown transfer kwarg '%s'" name k)
        "valid kwargs: from, to, fraction, count"
    ) unknown;
    let src = match List.assoc_opt "from" kwargs with
      | Some e -> resolve_comp_name ctx env e | None -> "?" in
    let dst = match List.assoc_opt "to" kwargs with
      | Some e -> resolve_comp_name ctx env e | None -> "?" in
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
    if not (Hashtbl.mem ctx.expanded_comp_tbl concrete
            || Hashtbl.mem ctx.comp_tbl comp) then
      Diagnostics.error ctx.diags
        ~code:"E265" ~loc
        ~message:(Printf.sprintf
          "intervention '%s' sets '%s' which is not a declared compartment"
          name concrete)
        ~hint:"check the compartments block, or fix the kwarg name \
               (e.g. fraction, count, from, to)"
        ();
    [Ir.Set { Ir.compartment = concrete; Ir.value = resolve_expr ctx env expr }]
  | AAdd (comp, idxs, expr) ->
    let idx_vals = List.map (index_item_to_str env) idxs in
    let concrete = if idx_vals = [] then comp
      else String.concat "_" (comp :: idx_vals) in
    [Ir.AddAction { Ir.add_compartment = concrete; Ir.add_count = resolve_expr ctx env expr }]

let expand_scheduled_actions ctx decls ~(kind : Ir.intervention_kind) =
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
    if iv.ivaction = [] then begin
      (* A scheduled intervention or event must carry at least one action.
         Empty was previously an implicit `ATransfer []` that misfired as E261. *)
      let noun = match kind with Ir.Scenario -> "intervention" | Ir.Event -> "event" in
      Diagnostics.error ctx.diags ~code:"E296" ~loc:iv_loc
        ~message:(Printf.sprintf "%s '%s' has no action" noun iv.ivname)
        ~hint:"add at least one action — a set (`S = S - 100`), an `add`, or a `transfer(...)`"
        ();
      []
    end else begin
    let base_name = if iv.ivindices = [] then None else Some iv.ivname in
    let combos = cartesian_product iv.ivindices ctx in
    List.filter_map (fun env ->
      let pass_guard = match iv.ivguard with
        | None   -> true
        | Some g -> eval_guard ctx env g
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
          (* Splice any `date_range(...)` calls into their expanded
             list of `EConst` entries before resolving. Phase 2 of the
             2026-05-22 typed-time proposal §4. *)
          let exprs = splice_date_ranges ctx exprs in
          (* gh#69: parametric `at [...]` lists must reach the runtime
             with their parameter references intact. Resolve every
             entry, then: if every entry is a compile-time constant,
             emit the legacy `Ir.AtTimes` form (existing goldens stay
             byte-identical); otherwise emit `Ir.AtTimesExpr` with the
             resolved IR expressions, which the Rust runtime evaluates
             once per simulation start against the current `params`
             vector. Mixed constant + parametric lists go through
             `AtTimesExpr` uniformly — constants become `Ir.Const`
             entries and the runtime evaluator handles them at zero
             cost. *)
          let resolved = List.map (fun e ->
            normalize_expr (resolve_expr ctx env e)
          ) exprs in
          let all_const = List.for_all (fun ir ->
            match ir with Ir.Const _ -> true | _ -> false
          ) resolved in
          if all_const then
            Ir.AtTimes (List.map (function
              | Ir.Const f -> f
              | _ -> assert false (* guarded by all_const *)
            ) resolved)
          else
            Ir.AtTimesExpr resolved
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
      let actions =
        List.concat_map
          (resolve_intervention_action ctx env ~name:iv_name ~loc:iv_loc)
          iv.ivaction
      in
      Some { Ir.name = iv_name; Ir.base_name; Ir.fire = Ir.Scheduled schedule;
             Ir.actions; Ir.kind }
    ) combos
    end
  ) decls

(* ── Reactive interventions (gh#204) ─────────────────────────────────────────
   Lower a reactive policy to an `Ir.intervention` with `fire = Reactive`. The
   trigger predicate is a dedicated ADT (not the shared expr): observed() /
   sum_observed() are recognised ONLY here, never in a rate. *)

let cmp_of_binop = function
  | Lt -> Ir.CmpLt | Le -> Ir.CmpLe | Gt -> Ir.CmpGt
  | Ge -> Ir.CmpGe | Eq -> Ir.CmpEq | Neq -> Ir.CmpNeq
  | _  -> Ir.CmpEq   (* unreachable: caller checks the op is a comparison *)

(* When the observed quantity is on the RIGHT (`2 <= observed(x)`), flip the
   operator so the lowered form is always `quantity <op> threshold`. *)
let flip_cmp = function
  | Ir.CmpLt -> Ir.CmpGt | Ir.CmpGt -> Ir.CmpLt
  | Ir.CmpLe -> Ir.CmpGe | Ir.CmpGe -> Ir.CmpLe
  | Ir.CmpEq -> Ir.CmpEq | Ir.CmpNeq -> Ir.CmpNeq

let is_comparison_binop = function
  | Lt | Le | Gt | Ge | Eq | Neq -> true | _ -> false

let is_observed_call = function
  | EFuncCall (("observed" | "sum_observed"), _) -> true
  | _ -> false

let lower_obs_quantity ctx env ~name ~loc (e : expr) : Ir.trigger_quantity =
  match e with
  | EFuncCall (fn, args) ->
    let stream_args = List.filter (fun (k, _) -> k = "") args in
    let window_arg  = List.assoc_opt "window" args in
    let stream = match stream_args with
      | [ (_, se) ] ->
        (match se with
         | EIdent (s, _) | EFuncCall (s, []) -> s
         | EIndex (base, items, _) ->
           let parts = List.map (index_item_to_str env) items in
           if parts = [] then base else String.concat "_" (base :: parts)
         | _ ->
           Diagnostics.error ctx.diags ~code:"E270" ~loc
             ~message:(Printf.sprintf
               "reactive intervention '%s': %s(...) argument must be an \
                observation stream name" name fn)
             ~hint:"e.g. observed(weekly_cases) or sum_observed(weekly_afp[p], window = 28 'days)"
             ();
           "?")
      | _ ->
        Diagnostics.error ctx.diags ~code:"E270" ~loc
          ~message:(Printf.sprintf
            "reactive intervention '%s': %s(...) needs exactly one stream \
             argument" name fn)
          ~hint:"observed(stream) or sum_observed(stream, window = ...)"
          ();
        "?"
    in
    (match fn with
     | "observed" ->
       (match window_arg with
        | Some _ ->
          Diagnostics.error ctx.diags ~code:"E271" ~loc
            ~message:(Printf.sprintf
              "reactive intervention '%s': observed(...) takes no `window`" name)
            ~hint:"use sum_observed(stream, window = ...) for a windowed sum"
            ()
        | None -> ());
       Ir.TQObserved { stream; window = None; reducer = Ir.RedLatest }
     | "sum_observed" ->
       let window = match window_arg with
         | Some w ->
           let wv = resolve_float_expr ctx w in
           if wv < 0.0 || not (Float.is_finite wv) then
             Diagnostics.error ctx.diags ~code:"E274" ~loc
               ~message:(Printf.sprintf
                 "reactive intervention '%s': `window` must be a non-negative \
                  finite duration (got %g)" name wv)
               ~hint:"e.g. window = 28 'days"
               ();
           Some wv
         | None ->
           Diagnostics.error ctx.diags ~code:"E271" ~loc
             ~message:(Printf.sprintf
               "reactive intervention '%s': sum_observed(...) requires a \
                `window = ...`" name)
             ~hint:"e.g. sum_observed(weekly_afp, window = 28 'days)"
             ();
           None
       in
       Ir.TQObserved { stream; window; reducer = Ir.RedSum }
     | _ -> assert false (* guarded by is_observed_call *))
  | _ -> assert false (* guarded by is_observed_call *)

let lower_threshold ctx env ~name ~loc (e : expr) : Ir.trigger_threshold =
  match resolve_expr ctx env e with
  | Ir.Const f -> Ir.TTConst f
  | Ir.Param p -> Ir.TTParam p
  | _ ->
    Diagnostics.error ctx.diags ~code:"E272" ~loc
      ~message:(Printf.sprintf
        "reactive intervention '%s': trigger threshold must be a constant or \
         a parameter" name)
      ~hint:"e.g. >= 2 or >= afp_trigger_threshold"
      ();
    Ir.TTConst 0.0

let lower_trigger_atom ctx env ~name ~loc (e : expr) : Ir.trigger_expr =
  match e with
  | EBinOp (op, lhs, rhs) when is_comparison_binop op ->
    let lobs = is_observed_call lhs and robs = is_observed_call rhs in
    if lobs && not robs then
      Ir.TECmp (lower_obs_quantity ctx env ~name ~loc lhs,
                cmp_of_binop op,
                lower_threshold ctx env ~name ~loc rhs)
    else if robs && not lobs then
      Ir.TECmp (lower_obs_quantity ctx env ~name ~loc rhs,
                flip_cmp (cmp_of_binop op),
                lower_threshold ctx env ~name ~loc lhs)
    else begin
      Diagnostics.error ctx.diags ~code:"E273" ~loc
        ~message:(Printf.sprintf
          "reactive intervention '%s': each trigger comparison must have \
           exactly one observed()/sum_observed() side" name)
        ~hint:"e.g. observed(weekly_cases) >= 10"
        ();
      Ir.TECmp (Ir.TQObserved { stream = "?"; window = None; reducer = Ir.RedLatest },
                Ir.CmpGe, Ir.TTConst 0.0)
    end
  | _ ->
    Diagnostics.error ctx.diags ~code:"E273" ~loc
      ~message:(Printf.sprintf
        "reactive intervention '%s': trigger predicate must be a comparison \
         (optionally combined with and/or/not)" name)
      ~hint:"e.g. when observed(weekly_cases) >= 10"
      ();
    Ir.TECmp (Ir.TQObserved { stream = "?"; window = None; reducer = Ir.RedLatest },
              Ir.CmpGe, Ir.TTConst 0.0)

let rec lower_trigger ctx env ~name ~loc (p : trig_pred) : Ir.trigger_expr =
  match p with
  | TgAnd (a, b) ->
    Ir.TEAnd (lower_trigger ctx env ~name ~loc a, lower_trigger ctx env ~name ~loc b)
  | TgOr (a, b) ->
    Ir.TEOr (lower_trigger ctx env ~name ~loc a, lower_trigger ctx env ~name ~loc b)
  | TgNot a -> Ir.TENot (lower_trigger ctx env ~name ~loc a)
  | TgAtom e -> lower_trigger_atom ctx env ~name ~loc e

let expand_reactive ctx decls =
  List.concat_map (fun (rx : reactive_decl) ->
    let rx_loc = diag_loc_of_ast_ctx ctx rx.rxloc in
    let base_name = if rx.rxindices = [] then None else Some rx.rxname in
    let combos = cartesian_product rx.rxindices ctx in
    List.filter_map (fun env ->
      let pass_guard = match rx.rxguard with
        | None   -> true
        | Some g -> eval_guard ctx env g
      in
      if not pass_guard then None
      else
      let parts = name_parts_from_bindings rx.rxindices env in
      let rx_name =
        if parts = [] then rx.rxname
        else rx.rxname ^ "_" ^ String.concat "_" parts
      in
      let when_ = lower_trigger ctx env ~name:rx_name ~loc:rx_loc rx.rxwhen in
      (* `after`, `cooldown`, and the `window` (below) are forward durations:
         a NaN/inf or negative value is a broken model, not a degenerate-but-
         valid one. Reject all three the same way (E274). *)
      let check_duration field v =
        if v < 0.0 || not (Float.is_finite v) then
          Diagnostics.error ctx.diags ~code:"E274" ~loc:rx_loc
            ~message:(Printf.sprintf
              "reactive intervention '%s': `%s` must be a non-negative finite \
               duration (got %g)" rx_name field v)
            ~hint:"durations are forward spans, e.g. after = 21 'days"
            ()
      in
      let after = match rx.rxafter with
        | None -> 0.0
        | Some e -> let a = resolve_float_expr ctx e in check_duration "after" a; a
      in
      let cooldown = match rx.rxcooldown with
        | None -> None
        | Some e -> let c = resolve_float_expr ctx e in check_duration "cooldown" c; Some c
      in
      let once = match rx.rxonce with
        | None -> true
        | Some (EIdent ("true", _))  | Some (EFuncCall ("true", []))  -> true
        | Some (EIdent ("false", _)) | Some (EFuncCall ("false", [])) -> false
        | Some _ ->
          Diagnostics.error ctx.diags ~code:"E275" ~loc:rx_loc
            ~message:(Printf.sprintf
              "reactive intervention '%s': `once` must be true or false" rx_name)
            ~hint:"once = true fires at most once; once = false allows repeats"
            ();
          true
      in
      (* once = true disables forever; cooldown rate-limits a REPEATING policy.
         The two are contradictory — reject rather than silently ignore one. *)
      if once && cooldown <> None then
        Diagnostics.error ctx.diags ~code:"E276" ~loc:rx_loc
          ~message:(Printf.sprintf
            "reactive intervention '%s': `once = true` and `cooldown` are \
             mutually exclusive" rx_name)
          ~hint:"once = true fires once and never again (drop cooldown); for a \
                 repeating rate-limited policy set once = false"
          ();
      let actions =
        resolve_intervention_action ctx env ~name:rx_name ~loc:rx_loc rx.rxaction
      in
      let trigger : Ir.reactive_trigger =
        { Ir.when_; Ir.after; Ir.once; Ir.cooldown }
      in
      Some { Ir.name = rx_name; Ir.base_name; Ir.fire = Ir.Reactive trigger;
             Ir.actions; Ir.kind = Ir.Scenario }
    ) combos
  ) decls

let expand_interventions ctx =
  expand_scheduled_actions ctx ctx.interv_decls ~kind:Ir.Scenario
  @ expand_scheduled_actions ctx ctx.event_decls ~kind:Ir.Event
  @ expand_reactive ctx ctx.reactive_decls

(* gh#204: every observation stream a reactive trigger reads must be a declared
   observation. Run as a post-pass once both interventions and observations are
   expanded (the expanded stream names — incl. stratified `weekly_cases_north` —
   are the source of truth), so an indexed trigger that resolves to a name no
   observation produces is caught. *)
let rec trigger_stream_refs (t : Ir.trigger_expr) : string list =
  match t with
  | Ir.TECmp (Ir.TQObserved { stream; _ }, _, _) -> [stream]
  | Ir.TEAnd (a, b) | Ir.TEOr (a, b) -> trigger_stream_refs a @ trigger_stream_refs b
  | Ir.TENot a -> trigger_stream_refs a

let validate_reactive_streams ctx (model : Ir.model) =
  let obs_names = List.map (fun (o : Ir.observation_model) -> o.Ir.name) model.Ir.observations in
  List.iter (fun (iv : Ir.intervention) ->
    match iv.Ir.fire with
    | Ir.Reactive t ->
      List.iter (fun stream ->
        if not (List.mem stream obs_names) then
          Diagnostics.error ctx.diags ~code:"E279" ~loc:Diagnostics.no_loc
            ~message:(Printf.sprintf
              "reactive intervention '%s': trigger references observation \
               stream '%s', which is not a declared observation" iv.Ir.name stream)
            ~hint:"declare it in observations { ... }, or fix the stream name"
            ()
      ) (trigger_stream_refs t.Ir.when_)
    | Ir.Scheduled _ -> ()
  ) model.Ir.interventions

(* ── Observation model expansion ─────────────────────────────────────────── *)

let expand_observations ctx =
  List.concat_map (fun od ->
    let od_loc = diag_loc_of_ast_ctx ctx od.oloc in
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
          | "columns" -> "add `columns { time : time, <value_col> : count }` — the explicit file schema"
          | "projection" -> "add `projected = incidence(<transition>)` or `projected = prevalence(<compartment>)`"
          | "measurement" -> "add `<value_col> ~ poisson(rate = ...)`, `neg_binomial(mean = ..., r = ...)`, etc."
          | _ -> "required field")
        ()
    in
    (* `emit_schedule` is OPTIONAL (§2.5): simulate-only, ignored in fit. *)
    let sched_v = od.oschedule in
    let proj_v = match od.oprojection with
      | Some p -> p
      | None -> missing_field "projection"; ProjIncidence (od.oname, [])
    in
    let meas_v = match od.omeasurement with
      | Some m -> m
      | None -> missing_field "measurement"; { om_scored = od.oname; om_lik = LikPoisson [("rate", EConst 1.0)] }
    in
    let lik_v = meas_v.om_lik in
    (* `columns { }` coherence checks (OCaml-side, internal — §2.2/§4.1).
       The file header is not seen here; only the declared schema. *)
    let columns_v = match od.ocolumns with
      | Some c -> c
      | None -> missing_field "columns"; []
    in
    (* A "real" measurement is one parsed from a `~` line (non-empty scored).
       The migration error path (`likelihood = ...`) yields an empty scored —
       its E273 already fired, so we don't pile on the scored/dead-column
       coherence checks (E276/E277), which would be spurious noise. *)
    let has_real_measurement = od.omeasurement <> None && meas_v.om_scored <> "" in
    let () =
      (* exactly one `: time` column *)
      let time_cols = List.filter (fun c -> c.oc_role = ColTime) columns_v in
      (match time_cols with
       | [] when od.ocolumns <> None ->
         Diagnostics.error ctx.diags ~code:"E275" ~loc:od_loc
           ~message:(Printf.sprintf
             "observation '%s': `columns { }` declares no `: time` column" od.oname)
           ~hint:"every stream needs exactly one time axis, e.g. `time : time`" ()
       | _ :: _ :: _ ->
         Diagnostics.error ctx.diags ~code:"E275" ~loc:od_loc
           ~message:(Printf.sprintf
             "observation '%s': `columns { }` declares %d `: time` columns; exactly one is allowed"
             od.oname (List.length time_cols))
           ~hint:"a stream has a single time axis" ()
       | _ -> ());
      (* the `~` LHS (scored) must be a declared value column *)
      let value_cols = List.filter_map (fun c ->
        match c.oc_role with ColValue _ -> Some c.oc_name | _ -> None) columns_v in
      if has_real_measurement && od.ocolumns <> None
         && not (List.mem meas_v.om_scored value_cols) then
        Diagnostics.error ctx.diags ~code:"E276" ~loc:od_loc
          ~message:(Printf.sprintf
            "observation '%s': the scored column '%s' (`%s ~ ...`) is not a declared value column"
            od.oname meas_v.om_scored meas_v.om_scored)
          ~hint:(Printf.sprintf
            "declare it in `columns { }`, e.g. `%s : count`" meas_v.om_scored) ();
      (* no dead value columns: every value column is the scored LHS or
         referenced by name on the `~` RHS *)
      let rhs_names =
        let rec names_of = function
          | EIdent (n, _) -> [n]
          | EIndex (n, items, _) -> n :: List.concat_map (function
              | IPosn e -> names_of e | INamed (_, e) -> names_of e) items
          | EBinOp (_, a, b) -> names_of a @ names_of b
          | EUnOp (_, e) -> names_of e
          | ECond (p, a, b) -> names_of p @ names_of a @ names_of b
          | EFuncCall (_, args) -> List.concat_map (fun (_, e) -> names_of e) args
          | ESum (_, _, _, e) -> names_of e
          | EList es -> List.concat_map names_of es
          | ERange (a, b) -> names_of a @ names_of b
          | EConst _ | EUnit _ | EObsAccess _ | ERunMember _ -> []
        in
        match meas_v.om_lik with
        | LikNegBinomial k | LikPoisson k | LikNormal k
        | LikBinomial k | LikBetaBinomial k | LikBernoulli k ->
          List.concat_map (fun (_, e) -> names_of e) k
      in
      if has_real_measurement && od.ocolumns <> None then
        List.iter (fun vc ->
          if vc <> meas_v.om_scored && not (List.mem vc rhs_names) then
            Diagnostics.error ctx.diags ~code:"E277" ~loc:od_loc
              ~message:(Printf.sprintf
                "observation '%s': value column '%s' is declared but never used \
                 (neither the scored outcome nor referenced in the likelihood)"
                od.oname vc)
              ~hint:"remove the dead column, or reference it in the `~` RHS" ()
        ) value_cols;
      (* `[p in dim]` ↔ `: dim` cross-check (§4.1). Every header index needs a
         `: dim` column; every `: dim` column needs a header index. *)
      let dim_cols = List.filter_map (fun c ->
        match c.oc_role with ColDim d -> Some d | _ -> None) columns_v in
      let header_dims = List.filter_map (function
        | IBind (_, d) -> Some d | _ -> None) od.oindices in
      if od.ocolumns <> None then begin
        List.iter (fun d ->
          if not (List.mem d dim_cols) then
            Diagnostics.error ctx.diags ~code:"E278" ~loc:od_loc
              ~message:(Printf.sprintf
                "observation '%s': header index `[_ in %s]` has no `%s : dim` column"
                od.oname d d)
              ~hint:(Printf.sprintf "declare `%s : dim` in `columns { }`" d) ()
        ) header_dims;
        List.iter (fun d ->
          if not (List.mem d header_dims) then
            Diagnostics.error ctx.diags ~code:"E278" ~loc:od_loc
              ~message:(Printf.sprintf
                "observation '%s': column `%s : dim` has no matching header index `[_ in %s]`"
                od.oname d d)
              ~hint:(Printf.sprintf
                "index the stream header, e.g. `%s[p in %s] { ... }`, or remove the column"
                od.oname d) ()
        ) dim_cols
      end
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
    let emit_schedule = match sched_v with
      | None -> None
      | Some (SchedEvery every) ->
        let step = resolve_float_expr ctx every in
        Some (Ir.ObsRegular { Ir.start = t_start; Ir.step; Ir.end_ = t_end })
      | Some (SchedAt ts) ->
        Some (Ir.ObsAtTimes (List.map (resolve_float_expr ctx) ts))
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
    (* `incidence(X)` with a bare (un-indexed) transition name. If X is a
       stratified transition family, the user means "sum over all strata,"
       symmetric to bare `prevalence` over a stratified compartment (§25.4).
       Emit CumulativeFlowSum over the expanded transition names when the
       family has >1 member; CumulativeFlow for the single/unstratified case.
       An unknown name falls through to CumulativeFlow so post-expansion
       Validate emits the clean E507 (unknown transition). *)
    let incidence_projection base =
      match expand_transition_name ctx base with
      | None         -> Ir.CumulativeFlow base
      | Some []      -> Ir.CumulativeFlow base
      | Some [single] -> Ir.CumulativeFlow single
      | Some many    ->
        (* Cross-strata aggregation gate (§5.2). A bare, un-indexed incidence
           projection over a STRATIFIED transition family on an UN-INDEXED
           stream would silently sum all strata and apply reporting uniformly.
           That decision must be explicit. (When the stream is itself indexed
           — `cases[p in patch]` — each cell resolves through the `EIndex`
           branch, never here; the explicit `sum(p in dim, ...)` forms parse as
           ESum and resolve through `ProjDerived`, also never here. So this
           fires precisely on the silent case.) *)
        if od.oindices = [] then begin
          Diagnostics.error ctx.diags
            ~code:"E280"
            ~loc:od_loc
            ~message:(Printf.sprintf
              "observation '%s' is un-indexed, but `incidence(%s)` would silently \
               sum all %d strata of '%s' and apply reporting uniformly"
              od.oname base (List.length many) base)
            ~hint:(Printf.sprintf
              "state the aggregation explicitly:\n\
              \  • uniform reporting:   <col> ~ ...( rho * sum(p in <dim>, incidence(%s[p])) )\n\
              \  • per-stratum reporting: <col> ~ ...( sum(p in <dim>, rho[p] * incidence(%s[p])) )"
              base base)
            ()
        end;
        Ir.CumulativeFlowSum many
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
         | Some (EIdent (n, _))    -> incidence_projection n
         | Some (EIndex (n, idxs, _)) ->
           Ir.CumulativeFlow (String.concat "_" (n :: List.map (index_item_to_str env) idxs))
         | _ -> Ir.CumulativeFlow "?")
      | ProjDerived (EFuncCall ("prevalence", args)) ->
        (match List.assoc_opt "" args with
         | Some (EIdent (n, _))    -> prevalence_projection n []
         | Some (EIndex (n, idxs, _)) ->
           prevalence_projection n (List.map (index_item_to_str env) idxs)
         | _ -> Ir.CurrentPop "?")
      | ProjDerived (EIdent (name, _) as e) ->
        (* Disambiguate: let-binding, compartment (prevalence), or
           transition (flow)? A let-bound bare identifier (e.g.
           `projected = I_total` with `let I_total = I_child + I_adult`)
           must inline the let body via the canonical resolver, not fall
           through to `CumulativeFlow "I_total"` — that name is neither a
           transition nor a compartment, so it would E507 (gh#164/#165). *)
        if Hashtbl.mem ctx.let_tbl name then
          Ir.DerivedExpr (resolve_expr ctx env e)
        else if Hashtbl.mem ctx.expanded_comp_tbl name then
          Ir.CurrentPop name
        else if Hashtbl.mem ctx.comp_tbl name then
          prevalence_projection name []
        else
          Ir.CumulativeFlow name
      | ProjDerived (EIndex (name, idxs, _)) ->
        let idx_vals = List.map (index_item_to_str env) idxs in
        let concrete = String.concat "_" (name :: idx_vals) in
        if Hashtbl.mem ctx.expanded_comp_tbl concrete then
          Ir.CurrentPop concrete
        else if Hashtbl.mem ctx.comp_tbl name then
          prevalence_projection name idx_vals
        else
          Ir.CumulativeFlow concrete
      (* EXPLICIT cross-strata incidence aggregation (§5.2):
         `sum(a in dim, incidence(tr[a]))` is the uniform-reporting form the
         aggregation gate (E280) directs the modeller to. It lowers to a
         CumulativeFlowSum over the per-level transitions — the SAME value the
         (now-rejected) bare `incidence(tr)` produced, but stated explicitly.
         The loop variable indexes the transition; each level instantiates one
         concrete flow. (A `sum` whose body is anything else falls through to
         the generic DerivedExpr arm below.) *)
      | ProjDerived (ESum (loop_var, dim, _, EFuncCall ("incidence", iargs)))
        when (match List.assoc_opt "" iargs with
              | Some (EIndex (_, _, _)) -> true | _ -> false) ->
        let inner = match List.assoc_opt "" iargs with
          | Some (EIndex (tr, idxs, _)) -> Some (tr, idxs) | _ -> None in
        (match inner with
         | Some (tr, idxs) ->
           let levels = match List.assoc_opt dim ctx.dim_registry with
             | Some ls -> ls | None -> [] in
           (* For each level, bind loop_var → level in a local env and resolve
              the transition's concrete name. *)
           let flows = List.map (fun lvl ->
             let local_env = (loop_var, lvl) :: env in
             let idx_vals = List.map (index_item_to_str local_env) idxs in
             String.concat "_" (tr :: idx_vals)
           ) levels in
           (match flows with
            | []       -> Ir.CumulativeFlow tr
            | [single] -> Ir.CumulativeFlow single
            | many     -> Ir.CumulativeFlowSum many)
         | None -> Ir.DerivedExpr (resolve_expr ctx env (ESum (loop_var, dim, None, EConst 0.0))))
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
    (* Declared value columns other than the scored outcome are the
       per-observation aux data the likelihood may reference by name (§3): the
       binomial denominator `n = tested`, a person-time offset, a reporting
       fraction. Register them so `resolve_expr` maps those identifiers to
       `Ir.ObsColumnRef` rather than E100. Scoped to this stream's likelihood
       resolution; cleared after. *)
    let aux_cols =
      List.filter_map (fun c ->
        match c.oc_role with
        | ColValue _ when c.oc_name <> meas_v.om_scored -> Some c.oc_name
        | _ -> None) columns_v
    in
    ctx.obs_aux_cols <- aux_cols;
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
    (* Each differentiable argument is a [diffable] with an EMPTY grad here; the
       obs/σ² autodiff driver (a later pass) populates the grads. `n` is a bare
       expr — θ-independent, no grad. *)
    let diff e : Ir.diffable = { Ir.expr = e; Ir.grad = []; Ir.proj_grad = None } in
    let likelihood = match lik_v with
      | LikNegBinomial kwargs ->
        Ir.NegBinomial {
          Ir.mean       = diff (resolve_kw kwargs "mean");
          Ir.dispersion = diff (resolve_kw kwargs "r");
        }
      | LikPoisson kwargs ->
        Ir.Poisson { Ir.rate = diff (resolve_kw kwargs "rate") }
      | LikNormal kwargs ->
        Ir.Normal {
          Ir.mean = diff (resolve_kw kwargs "mean");
          Ir.sd   = diff (resolve_kw kwargs "sd");
        }
      | LikBinomial kwargs ->
        Ir.Binomial {
          Ir.n = resolve_kw kwargs "n";
          Ir.p = diff (resolve_kw kwargs "p");
        }
      | LikBetaBinomial kwargs ->
        Ir.BetaBinomial {
          Ir.n     = resolve_kw kwargs "n";
          Ir.alpha = diff (resolve_kw kwargs "alpha");
          Ir.beta  = diff (resolve_kw kwargs "beta");
        }
      | LikBernoulli kwargs ->
        Ir.Bernoulli { Ir.p = diff (resolve_kw kwargs "p") }
    in
    ctx.obs_aux_cols <- [];
    let parts = name_parts_from_bindings od.oindices env in
    let obs_name =
      if parts = [] then od.oname
      else od.oname ^ "_" ^ String.concat "_" parts
    in
    (* Structured (dimension, level) selector for this expanded leaf — the
       by-name routing key the Rust long-form loader uses (§4.2). Shared
       top-level [stratum_of_bindings] (also used by generated quantities). An
       unstratified stream ([od.oindices = []]) yields []. *)
    let stratum = stratum_of_bindings od.oindices env in
    (* `from <label>` data-source key; defaults to the (unexpanded) stream
       name — every expanded leaf of a stratified stream shares the source. *)
    let source = match od.osource with Some s -> s | None -> od.oname in
    let ir_columns = List.map (fun c ->
      { Ir.col_name = c.oc_name;
        Ir.col_role = (match c.oc_role with
          | ColTime    -> Ir.RoleTime
          | ColDim d   -> Ir.RoleDim d
          | ColValue k -> Ir.RoleValue (ir_param_kind_of_ast k)); }
    ) columns_v in
    Some { Ir.name        = obs_name;
      Ir.obs_source     = source;
      Ir.columns       = ir_columns;
      Ir.scored        = meas_v.om_scored;
      Ir.emit_schedule = emit_schedule;
      Ir.stratum;
      Ir.projection;
      Ir.likelihood;
    }
    ) combos
  ) ctx.obs_decls

(* ── Generated quantities (proposal 2026-06-25) ───────────────────────────────
   The quantity classifier: per cell, decide whether a body is a temporal
   reduction, a bare series, or reduction arithmetic over earlier scalar
   quantities, and lower it to an [Ir.quantity_body]. Strata expand OUTER (like
   observations); the body is classified+resolved INNER with the per-cell env.

   A quantity name never enters ordinary name resolution, so a reference to a
   prior quantity in a body would otherwise hit E100 — the classifier detects
   quantity references up front (via [all_q_names]) and routes them through
   [ScalarExpr]/[QRef] instead. *)

type quantity_shape = QShScalar | QShSeries

(* Classify one cell of a quantity declaration. [cell_stratum] is this cell's
   (dim, level) tag; [all_q_names] is every quantity NAME in the file (for
   forward-reference detection); [declared] is the list of already-expanded
   PRIOR quantity leaves as (name, stratum, shape). Returns the lowered body and
   its shape, or [None] after emitting a diagnostic. *)
let classify_quantity_body ctx env
    (cell_stratum : (string * string) list)
    (all_q_names : string list)
    (declared : (string * (string * string) list * quantity_shape) list)
    (body : expr) (qd_loc : Diagnostics.loc)
    : (Ir.quantity_body * quantity_shape) option =
  let err ?hint msg =
    Diagnostics.error ctx.diags ~code:"E289" ~loc:qd_loc ~message:msg ?hint ();
    None
  in
  (* Located E288 for a directly-typed `dt` leaf anywhere in a State body — `dt`
     is the integrator step, meaningless when a quantity is read at output
     cadence. (ir::validate is the authoritative backstop for the transitive
     case; here we give the friendly located diagnostic for the direct case.) *)
  let rec find_dt = function
    | EIdent ("dt", l) -> Some l
    | EBinOp (_, a, b) -> (match find_dt a with Some l -> Some l | None -> find_dt b)
    | EUnOp (_, x) -> find_dt x
    | ECond (p, a, b) ->
      (match find_dt p with Some l -> Some l
       | None -> (match find_dt a with Some l -> Some l | None -> find_dt b))
    | EFuncCall (_, args) -> List.find_map (fun (_, e) -> find_dt e) args
    | EIndex (_, items, _) ->
      List.find_map (function IPosn e | INamed (_, e) -> find_dt e) items
    | ESum (_, _, _, e) -> find_dt e
    | EList es -> List.find_map find_dt es
    | ERange (a, b) -> (match find_dt a with Some l -> Some l | None -> find_dt b)
    | EConst _ | EUnit _ | EIdent _ | EObsAccess _ | ERunMember _ -> None
  in
  let check_dt e =
    match find_dt e with
    | Some l ->
      Diagnostics.error ctx.diags ~code:"E288" ~loc:(diag_loc_of_ast_ctx ctx l)
        ~message:"`dt` is only valid in a rate, not a quantity"
        ~hint:"a quantity is read at output cadence, where the integrator step \
               `dt` has no value"
        ();
      false
    | None -> true
  in
  (* Does an expression reference any quantity name (prior OR forward)? Drives
     the State-vs-Derived split and rejects a reduction applied to a scalar. *)
  let rec refs_quantity = function
    | EIdent (n, _) -> List.mem n all_q_names
    | EIndex (n, items, _) ->
      List.mem n all_q_names
      || List.exists (function IPosn e | INamed (_, e) -> refs_quantity e) items
    | EBinOp (_, a, b) -> refs_quantity a || refs_quantity b
    | EUnOp (_, x) -> refs_quantity x
    | ECond (p, a, b) -> refs_quantity p || refs_quantity a || refs_quantity b
    | EFuncCall (_, args) -> List.exists (fun (_, e) -> refs_quantity e) args
    | ESum (_, _, _, e) -> refs_quantity e
    | EList es -> List.exists refs_quantity es
    | ERange (a, b) -> refs_quantity a || refs_quantity b
    | EConst _ | EUnit _ | EObsAccess _ | ERunMember _ -> false
  in
  (* Does an expression contain an `observations.<stream>` access anywhere? An
     EObsAccess is meaningful ONLY as the whole quantity body or as the SOLE
     argument of a temporal reduction; nested in arithmetic, mixed with state,
     or used as a Derived/QRef operand it is malformed (E289). *)
  let rec contains_obs_access = function
    | EObsAccess _ -> true
    | EIndex (_, items, _) ->
      List.exists (function IPosn e | INamed (_, e) -> contains_obs_access e) items
    | EBinOp (_, a, b) -> contains_obs_access a || contains_obs_access b
    | EUnOp (_, x) -> contains_obs_access x
    | ECond (p, a, b) ->
      contains_obs_access p || contains_obs_access a || contains_obs_access b
    | EFuncCall (_, args) -> List.exists (fun (_, e) -> contains_obs_access e) args
    | ESum (_, _, _, e) -> contains_obs_access e
    | EList es -> List.exists contains_obs_access es
    | ERange (a, b) -> contains_obs_access a || contains_obs_access b
    | EConst _ | EUnit _ | EIdent _ | ERunMember _ -> false
  in
  (* Validate an `observations.<stream>` source at compile time: the stream must
     name a DECLARED observation, and (v1.1) it must be unstratified. Emits a
     located E289 and returns false on failure. *)
  let check_obs_stream stream sloc : bool =
    let loc = diag_loc_of_ast_ctx ctx sloc in
    match List.find_opt (fun (o : obs_decl) -> o.oname = stream) ctx.obs_decls with
    | None ->
      Diagnostics.error ctx.diags ~code:"E289" ~loc
        ~message:(Printf.sprintf
          "observations.%s: no observation stream '%s' is declared" stream stream)
        ~hint:"reference a stream named in the `observations { }` block"
        ();
      false
    | Some o when o.oindices <> [] ->
      Diagnostics.error ctx.diags ~code:"E289" ~loc
        ~message:(Printf.sprintf
          "observations.%s: a stratified observation source is not supported in \
           v1.1; reference an unstratified stream" stream)
        ~hint:"reduce an unstratified observation stream (no `[idx in dim]` \
               bindings on its declaration)"
        ();
      false
    | Some _ -> true
  in
  (* A temporal reduction folds a per-instant State series `inner`. The inner is
     resolved as an ordinary State expr; it must not itself reference a reduced
     scalar quantity (a reduction-on-a-reduction is malformed → E289). *)
  let reduced_state reduce inner =
    match inner with
    | EObsAccess (stream, sloc) ->
      (* v1.1: reduce the simulated observation series y_sim of `stream`. The
         reduction wraps a QSObservation source exactly as it wraps a State one. *)
      if check_obs_stream stream sloc then
        Some (Ir.QBReduced { source = Ir.QSObservation stream;
                             reduce = Some reduce },
              QShScalar)
      else None
    | _ when contains_obs_access inner ->
      err ~hint:"reduce a bare `observations.<stream>`; an observation source \
                 cannot be combined with arithmetic or latent state"
        "an observation source must be reduced on its own, not mixed into an \
         expression"
    | _ ->
      if refs_quantity inner then
        err ~hint:"a temporal reduction folds a per-instant series; a reduced \
                   scalar is already collapsed — combine scalars with arithmetic \
                   (e.g. `a - b`) instead"
          "cannot apply a temporal reduction to a reduced scalar quantity"
      else if not (check_dt inner) then None
      else
        Some (Ir.QBReduced { source = Ir.QSState (resolve_expr ctx env inner);
                             reduce = Some reduce },
              QShScalar)
  in
  (* Build a reduction-arithmetic ScalarExpr: leaves are prior scalar QRefs,
     params, or consts. A series QRef, forward QRef, cross-stratum QRef, or a
     mixed-in compartment/let is E289. *)
  (* Shared hint for the "series value mixed into reduction arithmetic" errors
     (compartment or `let`). Corrects the common misconception that lets are out
     of scope in quantities: they are in scope in *series* quantities; the clash
     here is a shape mismatch (whole-trajectory scalar vs per-instant series). *)
  let series_mix_hint =
    "`let` bindings and compartments ARE in scope in series quantities (a \
     quantity with no top-level reduction); they only clash here, mixed with a \
     reduced scalar, because the shapes differ. If the scalar factor is a model \
     constant (e.g. `let R0 = beta / gamma`), keep it a `let` so the whole \
     expression stays a series."
  in
  let scalar_leaf name =
    let matches = List.filter (fun (n, _, _) -> n = name) declared in
    if matches <> [] then
      (match List.find_opt (fun (_, s, _) -> s = cell_stratum) matches with
       | Some (_, _, QShScalar) ->
         Some (Ir.SQRef { Ir.qref_name = name; Ir.qref_stratum = cell_stratum })
       | Some (_, _, QShSeries) ->
         err (Printf.sprintf
           "quantity '%s' is a series (one value per snapshot); only reduced \
            scalars combine in reduction arithmetic" name)
       | None ->
         if List.exists (fun (_, _, sh) -> sh = QShSeries) matches then
           err (Printf.sprintf
             "quantity '%s' is a series; only reduced scalars combine in \
              reduction arithmetic" name)
         else
           err (Printf.sprintf
             "quantity '%s' is declared at a different stratum; reduction \
              arithmetic may only reference scalars of the same stratum" name))
    else if List.mem name all_q_names then
      err (Printf.sprintf
        "quantity '%s' is used before it is declared; reduction arithmetic may \
         only reference quantities declared earlier" name)
    else if Hashtbl.mem ctx.scalar_param_tbl name
            || is_expanded_indexed_param_name ctx name then
      Some (Ir.SParam name)
    else if Hashtbl.mem ctx.comp_tbl name || Hashtbl.mem ctx.expanded_comp_tbl name then
      err ~hint:series_mix_hint
        (Printf.sprintf
        "cannot combine compartment '%s' with reduced scalar quantities; a \
         per-instant state value and a whole-trajectory scalar have different \
         shapes" name)
    else if Hashtbl.mem ctx.let_tbl name then
      err ~hint:series_mix_hint
        (Printf.sprintf
        "cannot combine `let` binding '%s' with reduced scalar quantities; a \
         `let` is a per-instant series value and a reduced quantity is a \
         whole-trajectory scalar, so the shapes differ" name)
    else
      err (Printf.sprintf "unknown name '%s' in reduction arithmetic" name)
  in
  let rec build_scalar_expr e : Ir.scalar_expr option =
    match e with
    | EConst f -> Some (Ir.SConst f)
    | EUnit (f, u) -> Some (Ir.SConst (unit_to_model_time ctx f u))
    | EIdent (name, _) -> scalar_leaf name
    | EIndex (name, items, _) ->
      if List.mem name all_q_names then begin
        (* Indexed prior-scalar QRef. Pragmatic check: the explicit index levels
           must all belong to this cell's stratum (cross-stratum is rejected);
           the QRef then carries the cell stratum. *)
        let levels = List.map (index_item_to_str env) items in
        let cell_levels = List.map snd cell_stratum in
        if List.for_all (fun lv -> List.mem lv cell_levels) levels then
          scalar_leaf name
        else
          err (Printf.sprintf
            "quantity '%s' is referenced at a different stratum; reduction \
             arithmetic may only reference scalars of the same stratum" name)
      end else
        err (Printf.sprintf "cannot index '%s' in reduction arithmetic" name)
    | EBinOp (op, a, b) ->
      (match build_scalar_expr a, build_scalar_expr b with
       | Some sa, Some sb ->
         Some (Ir.SBinOp { op = ir_bin_op op; left = sa; right = sb })
       | _ -> None)
    | EUnOp (op, x) ->
      (match build_scalar_expr x with
       | Some sx -> Some (Ir.SUnOp { op = ir_un_op op; arg = sx })
       | None -> None)
    | ECond (p, a, b) ->
      (match build_scalar_expr p, build_scalar_expr a, build_scalar_expr b with
       | Some sp, Some sa, Some sb ->
         Some (Ir.SCond { pred = sp; then_ = sa; else_ = sb })
       | _ -> None)
    | EFuncCall (fn, _) ->
      err (Printf.sprintf
        "function '%s' is not allowed in reduction arithmetic; combine reduced \
         scalar quantities with +, -, *, / and comparisons only" fn)
    | EObsAccess (stream, _) ->
      err ~hint:"reduce a bare `observations.<stream>` (e.g. `max(observations.\
                 stream)`); an observation source is not a scalar operand"
        (Printf.sprintf
          "observations.%s cannot appear in reduction arithmetic" stream)
    | ERunMember { run; member; _ } ->
      (* Unreachable: the body-level `find_run_member` guard rejects any
         run-rooted reference (E293) before classification reaches here. Kept
         for exhaustiveness. *)
      err (Printf.sprintf
        "`%s....%s` is a contrast operand, not valid in a quantities recipe" run member)
    | ESum _ | EList _ | ERange _ ->
      err "this form is not allowed in reduction arithmetic"
  in
  let classify_non_reduction body =
    match body with
    | EObsAccess (stream, sloc) ->
      (* v1.1: an observation source must be reduced. A bare observation SERIES
         has its own observation-time axis (the stream's `emit_schedule`/fit
         leaves), distinct from the trajectory snapshot grid, so it cannot be
         rendered against the same time column as a state series. Reduce it.
         (`check_obs_stream` first, so a typo'd/stratified stream gets the more
         specific diagnostic.) *)
      if check_obs_stream stream sloc then
        err ~hint:(Printf.sprintf
          "wrap it in a temporal reduction, e.g. `max(observations.%s)`, \
           `integral(observations.%s)`, or `first_above(observations.%s, threshold)`"
          stream stream stream)
          (Printf.sprintf
            "observations.%s: a bare observation series is not supported in v1.1; an \
             observation source must be reduced (it has its own observation-time axis)"
            stream)
      else None
    | _ when contains_obs_access body ->
      err ~hint:"reference a bare `observations.<stream>`, optionally wrapped in \
                 a single temporal reduction; it cannot be combined with latent \
                 state, arithmetic, or another quantity"
        "an observation source cannot be combined with state, arithmetic, or a \
         quantity reference"
    | _ ->
      if refs_quantity body then
        (* Reduction arithmetic → Derived (a ScalarExpr over prior scalars). *)
        (match build_scalar_expr body with
         | Some se -> Some (Ir.QBDerived se, QShScalar)
         | None -> None)
      else
        (* A bare State series (e.g. `prevalence = I / N`). *)
        if not (check_dt body) then None
        else
          Some (Ir.QBReduced { source = Ir.QSState (resolve_expr ctx env body);
                               reduce = None },
                QShSeries)
  in
  let wrong_arity fn expected =
    err (Printf.sprintf "reduction '%s' takes %s" fn expected)
  in
  (* A run-rooted `<run>.<ns>.<member>` reference is a *contrast* operand; in a
     `quantities { }` recipe the run is implicit, so the prefix must be dropped
     (cross-context diagnostic, proposal 2026-06-25). Detect one anywhere in the
     body and emit the located fix before the ordinary classification runs. *)
  let rec find_run_member = function
    | ERunMember { run; ns; member; loc } -> Some (run, ns, member, loc)
    | EIndex (_, items, _) ->
      List.find_map (function IPosn e | INamed (_, e) -> find_run_member e) items
    | EBinOp (_, a, b) ->
      (match find_run_member a with Some r -> Some r | None -> find_run_member b)
    | EUnOp (_, x) -> find_run_member x
    | ECond (p, a, b) ->
      (match find_run_member p with Some r -> Some r
       | None -> (match find_run_member a with Some r -> Some r | None -> find_run_member b))
    | EFuncCall (_, args) -> List.find_map (fun (_, e) -> find_run_member e) args
    | ESum (_, _, _, e) -> find_run_member e
    | EList es -> List.find_map find_run_member es
    | ERange (a, b) ->
      (match find_run_member a with Some r -> Some r | None -> find_run_member b)
    | EConst _ | EUnit _ | EIdent _ | EObsAccess _ -> None
  in
  (match find_run_member body with
   | Some (run, ns_v, member, rloc) ->
     let ns = (match ns_v with NsQuantities -> "quantities" | NsObservations -> "observations") in
     (* The corrected in-recipe form differs by namespace: a quantity is
        referenced BARE (`quantities.foo` does not parse), whereas an observation
        series keeps its `observations.` prefix (`observations.afp` parses). *)
     let dropped, suggested = match ns_v with
       | NsQuantities   -> Printf.sprintf "%s.quantities." run, member
       | NsObservations -> Printf.sprintf "%s." run, Printf.sprintf "observations.%s" member
     in
     Diagnostics.error ctx.diags ~code:"E293"
       ~loc:(diag_loc_of_ast_ctx ctx rloc)
       ~message:(Printf.sprintf
         "`%s.%s.%s` is a contrast operand (a run-prefixed reference), not valid \
          in a `quantities { }` recipe" run ns member)
       ~hint:(Printf.sprintf
         "in a quantities recipe the run is implicit — drop the `%s` prefix and \
          write `%s`; run-prefixed references belong in a `contrasts { }` block"
         dropped suggested)
       ();
     None
   | None ->
  match body with
  | EFuncCall (("total" | "sum") as fn, _) ->
    err ~hint:"summing a stock over snapshots is cadence-dependent; cumulative \
               sums arrive with the flow source in a later increment"
      (Printf.sprintf "`%s(...)` is not available in v1 quantities" fn)
  | EFuncCall (fn, args)
    when is_temporal_reduction_name fn || fn = "max" || fn = "min" ->
    let one = (match args with [("", x)] -> Some x | _ -> None) in
    let two = (match args with [("", a); ("", b)] -> Some (a, b) | _ -> None) in
    (match fn with
     | "final" ->
       (match one with Some x -> reduced_state (Ir.RValue Ir.VFinal) x
        | None -> wrong_arity fn "one argument, e.g. final(D)")
     | "mean" ->
       (match one with Some x -> reduced_state (Ir.RValue Ir.VMean) x
        | None -> wrong_arity fn "one argument, e.g. mean(I)")
     | "integral" ->
       (match one with Some x -> reduced_state Ir.RIntegral x
        | None -> wrong_arity fn "one argument, e.g. integral(I)")
     | "time_of_max" ->
       (match one with Some x -> reduced_state (Ir.RTime Ir.TimeOfMax) x
        | None -> wrong_arity fn "one argument, e.g. time_of_max(I)")
     | "time_of_min" ->
       (match one with Some x -> reduced_state (Ir.RTime Ir.TimeOfMin) x
        | None -> wrong_arity fn "one argument, e.g. time_of_min(I)")
     | "max" ->
       (* unary max → temporal VMax; binary max(a,b) is pointwise → State. *)
       (match one with Some x -> reduced_state (Ir.RValue Ir.VMax) x
        | None -> classify_non_reduction body)
     | "min" ->
       (match one with Some x -> reduced_state (Ir.RValue Ir.VMin) x
        | None -> classify_non_reduction body)
     | "count_above" ->
       (match two with
        | Some (x, th) ->
          reduced_state (Ir.RValue (Ir.VCountAbove (resolve_expr ctx env th))) x
        | None -> wrong_arity fn "two arguments, e.g. count_above(I, thresh)")
     | "count_below" ->
       (match two with
        | Some (x, th) ->
          reduced_state (Ir.RValue (Ir.VCountBelow (resolve_expr ctx env th))) x
        | None -> wrong_arity fn "two arguments, e.g. count_below(I, thresh)")
     | "first_above" ->
       (match two with
        | Some (x, th) ->
          reduced_state (Ir.RTime (Ir.FirstAbove (resolve_expr ctx env th))) x
        | None -> wrong_arity fn "two arguments, e.g. first_above(I, thresh)")
     | "first_below" ->
       (match two with
        | Some (x, th) ->
          reduced_state (Ir.RTime (Ir.FirstBelow (resolve_expr ctx env th))) x
        | None -> wrong_arity fn "two arguments, e.g. first_below(I, thresh)")
     | "last_above" ->
       (match two with
        | Some (x, th) ->
          reduced_state (Ir.RTime (Ir.LastAbove (resolve_expr ctx env th))) x
        | None -> wrong_arity fn "two arguments, e.g. last_above(I, thresh)")
     | "last_below" ->
       (match two with
        | Some (x, th) ->
          reduced_state (Ir.RTime (Ir.LastBelow (resolve_expr ctx env th))) x
        | None -> wrong_arity fn "two arguments, e.g. last_below(I, thresh)")
     | _ -> classify_non_reduction body)
  | _ -> classify_non_reduction body)

(* Expand the `quantities { }` blocks into fully-expanded [Ir.quantity] leaves.
   Strata expand OUTER (cartesian_product over the decl's index bindings);
   the body is classified INNER per cell. A declaration's own cells are
   registered as PRIOR only AFTER the whole declaration is expanded, so a
   stratified decl's sibling cells are not visible to one another's QRefs. *)
let expand_quantities ctx =
  let all_q_names =
    List.map (fun (qd : quantity_decl) -> qd.qd_name) ctx.quantity_decls in
  let seen = Hashtbl.create 16 in   (* base names declared so far (collision) *)
  let declared
    : (string * (string * string) list * quantity_shape) list ref = ref [] in
  List.concat_map (fun (qd : quantity_decl) ->
    let qd_loc = diag_loc_of_ast_ctx ctx qd.qd_loc in
    let name = qd.qd_name in
    let collision =
      if Hashtbl.mem ctx.comp_tbl name || Hashtbl.mem ctx.expanded_comp_tbl name
      then Some "compartment"
      else if Hashtbl.mem ctx.scalar_param_tbl name
              || is_expanded_indexed_param_name ctx name
              || is_indexed_param ctx name
      then Some "parameter"
      else if Hashtbl.mem ctx.let_tbl name then Some "let binding"
      else if Hashtbl.mem ctx.func_tbl name then Some "forcing function"
      else if List.exists (fun (o : obs_decl) -> o.oname = name) ctx.obs_decls
      then Some "observation stream"
      else if Hashtbl.mem seen name then Some "earlier quantity"
      else None
    in
    match collision with
    | Some kind ->
      Diagnostics.error ctx.diags ~code:"E289" ~loc:qd_loc
        ~message:(Printf.sprintf
          "quantity '%s' collides with a %s of the same name" name kind)
        ~hint:"give the quantity a distinct name; to report the colliding \
               value, reference it in the body under the new name \
               (e.g. `R0_hat = R0` reports the `let R0` as `R0_hat`)"
        ();
      []
    | None ->
      Hashtbl.add seen name ();
      let combos = cartesian_product qd.qd_indices ctx in
      (* Classify each cell; defer registering into [declared] until the whole
         declaration is done (sibling cells are not prior to one another). *)
      let cells = List.filter_map (fun env ->
        let cell_stratum = stratum_of_bindings qd.qd_indices env in
        match
          classify_quantity_body ctx env cell_stratum all_q_names !declared
            qd.qd_body qd_loc
        with
        | Some (q_body, shape) ->
          Some ({ Ir.q_name = name; Ir.q_stratum = cell_stratum; Ir.q_body;
                  (* Resolved dimension (#5) is filled by the dimcheck write-back
                     in [Compiler.finish_compile]; the expander has no dimension
                     inference, so it leaves it unset here. *)
                  Ir.q_dimension = None },
                shape)
        | None -> None
      ) combos in
      List.iter (fun ((leaf : Ir.quantity), shape) ->
        declared := (leaf.Ir.q_name, leaf.Ir.q_stratum, shape) :: !declared
      ) cells;
      List.map fst cells
  ) ctx.quantity_decls

(* ── Counterfactual contrasts (proposal 2026-06-25) ───────────────────────────
   Resolve each `contrasts { }` entry: two-sided name resolution of every
   run-rooted operand (run ∈ declared scenarios ∪ {fitted}; member ∈ the named
   sub-namespace) and lowering of the arithmetic body to [Ir.contrast_expr].
   There is no window: the counterfactual fork is *derived* in the reducer (the
   last saved snapshot before the toggled intervention's fire time) and the
   result is shaped over `[fork, run-end]`. Dimensional agreement of the operands
   is checked separately by `dimcheck` (the verify path). *)
let expand_contrasts ctx : Ir.contrast list =
  let scenario_names = List.map (fun (sd : scenario_decl) -> sd.scname) ctx.scenario_decls in
  let quantity_names = List.map (fun (qd : quantity_decl) -> qd.qd_name) ctx.quantity_decls in
  let obs_names      = List.map (fun (o : obs_decl) -> o.oname) ctx.obs_decls in
  let ns_str = function NsQuantities -> "quantities" | NsObservations -> "observations" in
  (* Resolve one run-rooted operand to Ir.CRunMember, emitting located,
     two-sided diagnostics for an undeclared run or member. *)
  let resolve_run_member run ns member rloc : Ir.contrast_expr option =
    let loc = diag_loc_of_ast_ctx ctx rloc in
    let run_ok = run = "fitted" || List.mem run scenario_names in
    let member_ok = match ns with
      | NsQuantities   -> List.mem member quantity_names
      | NsObservations -> List.mem member obs_names in
    if not run_ok then begin
      Diagnostics.error ctx.diags ~code:"E294" ~loc
        ~message:(Printf.sprintf
          "contrast operand `%s.%s.%s`: no run named '%s'"
          run (ns_str ns) member run)
        ~hint:"a contrast run is a declared `scenarios { }` preset, or the \
               reserved `fitted` (the no-overlay fitted model)"
        ();
      None
    end else if not member_ok then begin
      let kind, where = match ns with
        | NsQuantities   -> "quantity", "a `quantities { }` entry"
        | NsObservations -> "observation stream", "an `observations { }` stream" in
      Diagnostics.error ctx.diags ~code:"E294" ~loc
        ~message:(Printf.sprintf
          "contrast operand `%s.%s.%s`: no %s named '%s'"
          run (ns_str ns) member kind member)
        ~hint:(Printf.sprintf "`%s.%s` must name %s" (ns_str ns) member where)
        ();
      None
    end else
      let ir_ns = match ns with
        | NsQuantities -> Ir.NsQuantities
        | NsObservations -> Ir.NsObservations in
      Some (Ir.CRunMember { run; ns = ir_ns; member })
  in
  (* Lower the contrast arithmetic body. v1 grammar: run-rooted operands combined
     by `+ - * /`. A bare const, comparison, unary op, or any other form is a
     located error (a contrast operand must be a run-rooted reference). *)
  let rec lower_body cd_loc (e : expr) : Ir.contrast_expr option =
    match e with
    | ERunMember { run; ns; member; loc } -> resolve_run_member run ns member loc
    | EBinOp ((Add | Sub | Mul | Div) as op, a, b) ->
      (match lower_body cd_loc a, lower_body cd_loc b with
       | Some l, Some r -> Some (Ir.CBinOp { op = ir_bin_op op; left = l; right = r })
       | _ -> None)
    | _ ->
      Diagnostics.error ctx.diags ~code:"E295" ~loc:(diag_loc_of_ast_ctx ctx cd_loc)
        ~message:"a contrast body must combine run-rooted operands with + - * /"
        ~hint:"each operand is `<run>.quantities.<q>` or `<run>.observations.<stream>`; \
               reduce a series to a scalar in `quantities { }`, then contrast the \
               named quantity (v1 takes no inline reducer, const, or comparison here)"
        ();
      None
  in
  (* Contrast names must be unique: each lowers to one `contrasts/<name>.tsv`, so
     a duplicate would silently clobber its sibling. Reject the collision with a
     located error (mirrors the quantity name-collision check, E289). *)
  let seen : (string, unit) Hashtbl.t = Hashtbl.create 16 in
  List.filter_map (fun (cd : contrast_decl) ->
    if Hashtbl.mem seen cd.cd_name then begin
      Diagnostics.error ctx.diags ~code:"E298"
        ~loc:(diag_loc_of_ast_ctx ctx cd.cd_loc)
        ~message:(Printf.sprintf
          "duplicate contrast '%s' — two `contrasts { }` entries share this name"
          cd.cd_name)
        ~hint:"each contrast is written to `contrasts/<name>.tsv`, so names must \
               be unique; rename one of the entries"
        ();
      None
    end else begin
      Hashtbl.add seen cd.cd_name ();
      match lower_body cd.cd_loc cd.cd_body with
      | Some c_body -> Some { Ir.c_name = cd.cd_name; Ir.c_body }
      | None -> None
    end
  ) ctx.contrast_decls

(* ── Hierarchical-prior cycle / self-reference check ─────────────────────── *)

(** Collect the set of parameter names referenced anywhere inside an
    AST expression. Used by the cycle detector. *)
let rec collect_param_refs known_params acc = function
  | EConst _ | EUnit _ -> acc
  | EIdent (name, _) when List.mem name known_params -> name :: acc
  | EIdent (_, _) -> acc
  | EIndex (_, items, _) ->
    List.fold_left (fun a item ->
      match item with
      | IPosn e | INamed (_, e) -> collect_param_refs known_params a e
    ) acc items
  | EBinOp (_, l, r) ->
    let a = collect_param_refs known_params acc l in
    collect_param_refs known_params a r
  | EUnOp (_, e) -> collect_param_refs known_params acc e
  | ESum (_, _, _, body) -> collect_param_refs known_params acc body
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
  | EObsAccess _ -> acc
  | ERunMember _ -> acc

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

(* ── Origin / shadowing checks ────────────────────────────────────────────── *)

(** M14 (gh#98): validate the top-level `origin = date("...")` string up
    front, before any `origin_rata_die` / date conversion derives values
    from it. A malformed or out-of-range origin previously produced a
    nonsense `origin_rata_die` (and garbage internal-time conversions for
    every `date()` literal) with no diagnostic. Emit a named E223. *)
let check_origin ctx =
  match ctx.origin with
  | None -> ()
  | Some s ->
    (match parse_iso_date s with
     | Ok _ -> ()
     | Error msg ->
       Diagnostics.error ctx.diags ~code:"E223" ~loc:Diagnostics.no_loc
         ~message:(Printf.sprintf "origin: %s" msg)
         ~hint:"origin must be a calendar date YYYY-MM-DD with month \
                01..12 and a day valid for that month (leap-aware)"
         ())

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

(** E283: reject a `sum` bound variable that shadows an enclosing index or
    bound variable. Resolution is first-match-wins (`resolve_expr`'s ESum arm
    prepends `(v, _) :: env`), so a shadowing `sum` silently rebinds — e.g.
    `sum(p in patch, …)` inside `infection[p in patch]` becomes a global sum
    over all patches instead of the per-stratum term, with no diagnostic. This
    is a silent-wrong result, so it is a hard error. Checked across every
    index-binding construct that carries a user expression: transitions, lets,
    init, observations, interventions, events, and forcing args. (ODE equations
    and the balance expr have no index binder, so they are exempt.) *)
let check_no_shadowing ctx =
  let report decl v =
    Diagnostics.error ctx.diags
      ~code:"E283"
      ~loc:Diagnostics.no_loc
      ~message:(Printf.sprintf
        "%s: sum variable '%s' shadows an enclosing binding of '%s'. \
         First-match-wins resolution would silently rebind it (turning a \
         per-stratum term into a global sum). Rename the sum variable."
        decl v v)
      ()
  in
  let rec walk decl bound (e : expr) =
    match e with
    | EConst _ | EUnit _ | EIdent _ -> ()
    | EIndex (_, items, _) ->
      List.iter (function IPosn e | INamed (_, e) -> walk decl bound e) items
    | EBinOp (_, l, r) -> walk decl bound l; walk decl bound r
    | EUnOp (_, e) -> walk decl bound e
    | ESum (v, _, _, b) ->
      if List.mem v bound then report decl v;
      walk decl (v :: bound) b
    | ECond (p, t, f) -> walk decl bound p; walk decl bound t; walk decl bound f
    | EFuncCall (_, args) -> List.iter (fun (_, e) -> walk decl bound e) args
    | EList es            -> List.iter (walk decl bound) es
    | ERange (lo, hi)     -> walk decl bound lo; walk decl bound hi
    | EObsAccess _        -> ()
    | ERunMember _        -> ()
  in
  List.iter (fun (tr : transition_decl) ->
    let decl = Printf.sprintf "transition '%s'" tr.trname in
    let seed = loop_vars_of_indices tr.trindices in
    List.iter (walk decl seed) (trans_dynamics_exprs tr.trdyn);
    match tr.trdst with
    | DstBranch branches -> List.iter (fun (_, w) -> walk decl seed w) branches
    | DstSum _ -> ()
  ) ctx.transitions;
  List.iter (fun (lb : let_binding) ->
    walk (Printf.sprintf "let '%s'" lb.lname)
      (loop_vars_of_indices lb.lindices) lb.lbody
  ) ctx.let_bindings;
  List.iter (fun (ie : init_entry) ->
    walk (Printf.sprintf "init '%s'" ie.icomp)
      (loop_vars_of_indices ie.ibindings) ie.ivalue
  ) ctx.init_entries;
  List.iter (fun (od : obs_decl) ->
    let decl = Printf.sprintf "observation '%s'" od.oname in
    let seed = loop_vars_of_indices od.oindices in
    (match od.oprojection with
     | Some (ProjDerived e) -> walk decl seed e
     | Some (ProjIncidence _) | Some (ProjPrevalence _) | None -> ());
    (match od.omeasurement with
     | Some om ->
       let kwargs = match om.om_lik with
         | LikNegBinomial a | LikPoisson a | LikNormal a
         | LikBinomial a | LikBetaBinomial a | LikBernoulli a -> a
       in
       List.iter (fun (_, e) -> walk decl seed e) kwargs
     | None -> ())
  ) ctx.obs_decls;
  let walk_action decl seed = function
    | ATransfer kwargs -> List.iter (fun (_, e) -> walk decl seed e) kwargs
    | ASet (_, _, e) | AAdd (_, _, e) -> walk decl seed e
  in
  (* interventions and events share intervention_decl (the same [p in dim]
     index binder + expr actions). *)
  List.iter (fun (iv : intervention_decl) ->
    List.iter
      (walk_action (Printf.sprintf "intervention '%s'" iv.ivname)
         (loop_vars_of_indices iv.ivindices)) iv.ivaction
  ) ctx.interv_decls;
  List.iter (fun (ev : intervention_decl) ->
    List.iter
      (walk_action (Printf.sprintf "event '%s'" ev.ivname)
         (loop_vars_of_indices ev.ivindices)) ev.ivaction
  ) ctx.event_decls;
  List.iter (fun (fd : func_decl) ->
    let decl = Printf.sprintf "forcing '%s'" fd.fname in
    let seed = loop_vars_of_indices fd.findices in
    List.iter (fun (_, e) -> walk decl seed e) fd.fargs
  ) ctx.func_decls

(** W105: a transition indexed by two levels of the SAME dimension where one of
    those index variables appears only in the rate (not the source/destination
    stoichiometry) is the per-(p,q) coupling antipattern — it generates P²−P
    transitions, each with its own flow accumulator. The intended form is one
    transition per stratum with a summed rate, `sum(q in dim where …, …)`. Warn
    (the per-pair form is legal — someone may genuinely want per-pair flows). *)
let check_quadratic_coupling ctx =
  let rec mentions v = function
    | EConst _ | EUnit _ -> false
    | EIdent (n, _) -> n = v
    | EIndex (n, items, _) ->
      n = v || List.exists (function IPosn e | INamed (_, e) -> mentions v e) items
    | EBinOp (_, l, r) -> mentions v l || mentions v r
    | EUnOp (_, e) -> mentions v e
    | ESum (sv, _, g, b) ->
      if sv = v then false   (* inner sum rebinds v (E283 forbids); stop *)
      else (match g with Some g -> guard_mentions v g | None -> false) || mentions v b
    | ECond (p, t, f) -> mentions v p || mentions v t || mentions v f
    | EFuncCall (_, args) -> List.exists (fun (_, e) -> mentions v e) args
    | EList es -> List.exists (mentions v) es
    | ERange (lo, hi) -> mentions v lo || mentions v hi
    | EObsAccess _ -> false
    | ERunMember _ -> false
  and guard_mentions v = function
    | GEq (a, b) | GNeq (a, b) -> a = v || b = v
    | GTab (_, idxs, _, operand) ->
      List.mem v idxs || (match operand with GoName n -> n = v | GoNum _ -> false)
    | GAnd (g1, g2) | GOr (g1, g2) -> guard_mentions v g1 || guard_mentions v g2
  in
  let stoich_ref_vars ((_, items) : stoich_ref) =
    List.filter_map (function
      | IPosn (EIdent (v, _)) | INamed (_, EIdent (v, _)) -> Some v
      | _ -> None) items
  in
  List.iter (fun (tr : transition_decl) ->
    let dims = List.filter_map (function
      | IBind (v, d) | IConsec (v, _, d) -> Some (v, d)
      | IComp _ -> None) tr.trindices in
    let count_dim d = List.length (List.filter (fun (_, d') -> d' = d) dims) in
    let stoich_vars =
      List.concat_map stoich_ref_vars tr.trsrc
      @ (match tr.trdst with
         | DstSum refs   -> List.concat_map stoich_ref_vars refs
         | DstBranch brs -> List.concat_map (fun (r, _) -> stoich_ref_vars r) brs)
    in
    let offending = List.exists (fun (v, d) ->
      count_dim d >= 2 && not (List.mem v stoich_vars)
      && List.exists (mentions v) (trans_dynamics_exprs tr.trdyn)
    ) dims in
    if offending then
      Diagnostics.warning ctx.diags ~code:"W105" ~loc:Diagnostics.no_loc
        ~message:(Printf.sprintf
          "transition '%s' is indexed by two levels of the same dimension where an \
           index appears only in the rate, not the stoichiometry. This generates one \
           transition per pair (O(P^2) transitions and flow columns). For coupling, \
           write one transition per stratum with a summed rate, e.g. \
           `sum(q in dim where dist[p,q] < r, ...)`." tr.trname) ()
  ) ctx.transitions

(* ── Surface time-typing (Phase 1 of typed-time proposal) ────────────────── *)

(** Surface-level time-typing pass. Implements rules from the
    2026-05-22 typed-time-and-dsl-ergonomics proposal at the AST
    level — before unit literals get scaled out by `resolve_expr`.

    Rules emitted here:
      Rule 1: E3xx on `Instant ± CalendarDuration` in DSL constant
              positions and in let-bindings whose body laundered
              through this shape.
      Rule 2: E3xx on `time_unit = 'months/'years` when
              `origin = date(...)` is declared.
      Rule 4: E3xx on bare-numeric entries inside `on=[...]` of a
              periodic forcing in anchored mode. (This rule is
              applied at the periodic-forcing expansion site, not
              here — see `expand_time_function_one`.)
      Rule 5: W3xx on bare-numeric `simulate.from`/`simulate.to`
              in anchored mode, and on bare-numeric `at [k, ...]`
              entries in intervention/event schedules.
      Rule 7: Extension of Rule 1 to recurring-schedule cadences
              (`every`, `from`, `until`). In anchored mode, a
              calendar-classified duration in any of those is an
              E3xx.

    All anchored-mode rules are vacuous when no `origin` is
    declared (proposal §1.1, decision of record). The unanchored
    dacca-style configuration — `time_unit = 'months` + per-month
    rate params — flows through untouched. *)
let check_surface_time_typing ctx =
  let anchored = ctx.origin <> None in
  let env = Time_typing.env_of_ctx
    ~let_tbl:ctx.let_tbl
    ~param_decls:ctx.param_decls
    ~origin_set:anchored
  in

  (* ── Rule 2: time_unit = 'months/'years with origin declared ──────── *)
  (if anchored then
    match ctx.time_unit with
    | Months | Years ->
      Diagnostics.error ctx.diags
        ~code:"E320"
        ~loc:Diagnostics.no_loc
        ~message:(Printf.sprintf
          "`time_unit = '%s` cannot be combined with `origin = date(\"...\")` \
           — the date/number conversion would drift because a calendar \
           %s is not a constant number of days"
          (unit_lit_to_string ctx.time_unit)
          (match ctx.time_unit with Months -> "month" | Years -> "year" | _ -> ""))
        ~hint:Time_typing.hint_time_unit_months_with_origin
        ()
    | _ -> ());

  (* ── Rule 1 / Rule 7: detect calendar-duration sinks in expressions ── *)
  (* We walk every relevant AST expression in the model. In
     anchored mode any `Instant ± Calendar` triggers E321 (Rule 1)
     and any calendar cadence in `every`/`from`/`until` triggers
     E322 (Rule 7). In unanchored mode nothing fires.

     We don't recurse into rate expressions (which never carry
     `Instant`s — the AST disallows them indirectly because
     compartment refs and rate params can't be Instants). We do
     descend into all kinds of expressions in the model that have
     constant-position semantics: bounds, simulate, init values,
     observation `every`/`at`, scheduled events/interventions,
     transition rates (just in case a future model puts a date()
     there), table expressions, ODE derivatives, etc. *)

  let walk_expr_rule1 ~loc ~context e =
    if anchored then
      Time_typing.walk_rule1 env e ~on_hit:(fun ~lhs ~rhs ->
        (* Pick the calendar-classed side for the error message. *)
        let cl = Time_typing.classify env lhs in
        let cr = Time_typing.classify env rhs in
        let bad = match cl, cr with
          | Time_typing.TCalendar, _ -> lhs
          | _, Time_typing.TCalendar -> rhs
          | _ -> rhs  (* shouldn't happen if walk_rule1 fired *)
        in
        Diagnostics.error ctx.diags
          ~code:"E321"
          ~loc
          ~message:(Printf.sprintf
            "calendar duration `%s` cannot translate an instant in %s"
            (Time_typing.show_short bad) context)
          ~hint:Time_typing.hint_calendar_plus_instant
          ())
  in

  (* Helper: classify a duration-typed expression and fire E322 if
     it ends up Calendar (used for Rule 7: every/from/until). *)
  let check_recurring_cadence ~loc ~field e =
    if anchored then begin
      match Time_typing.classify env e with
      | Time_typing.TCalendar ->
        Diagnostics.error ctx.diags
          ~code:"E322"
          ~loc
          ~message:(Printf.sprintf
            "calendar duration `%s` in recurring schedule `%s = ...`"
            (Time_typing.show_short e) field)
          ~hint:Time_typing.hint_calendar_cadence_in_recurring
          ()
      | _ -> ()
    end
  in

  (* Helper: warn on bare numeric in a time position (W324 for
     simulate, W325 for `at [...]` schedules). *)
  let is_bare_numeric (e : expr) : bool =
    match e with
    | EConst _ -> true
    | EUnOp (Neg, EConst _) -> true
    | _ -> false
  in
  let warn_bare_numeric ~loc ~code ~field ~hint e =
    if anchored && is_bare_numeric e then
      Diagnostics.warning ctx.diags
        ~code
        ~loc
        ~message:(Printf.sprintf
          "`%s = %s` is a bare number in a time position with \
           `origin = date(...)` declared — interpreted as \
           internal-time units from origin"
          field (Time_typing.show_short e))
        ~hint
        ()
  in

  (* ── transition rates ───────────────────────────────────────────────── *)
  List.iter (fun (tr : transition_decl) ->
    List.iter
      (walk_expr_rule1 ~loc:(diag_loc_of_ast_ctx ctx tr.trloc)
         ~context:(Printf.sprintf "transition '%s'" tr.trname))
      (trans_dynamics_exprs tr.trdyn)
  ) ctx.transitions;

  (* ── ODE derivatives ────────────────────────────────────────────────── *)
  List.iter (fun (od : ode_decl) ->
    walk_expr_rule1 ~loc:Diagnostics.no_loc
      ~context:(Printf.sprintf "ODE d(%s)/dt" od.ocomp) od.oderiv
  ) ctx.ode_decls;

  (* ── parameter bounds & priors ──────────────────────────────────────── *)
  (* Bounds: classify each bound expression. Per the proposal
     §3.3.2 invariant, a bound's classifier doesn't leak to uses
     of the parameter — but a bound that itself contains `date(...) +
     <calendar>` is genuinely malformed and should fire Rule 1. *)
  List.iter (fun pd ->
    let ploc = match pd with PScalar s -> s.ploc | PIndexed s -> s.ploc in
    let loc = diag_loc_of_ast_ctx ctx ploc in
    let pbounds = match pd with PScalar s -> s.pbounds | PIndexed s -> s.pbounds in
    (match pbounds with
     | Some (lo, hi) ->
       walk_expr_rule1 ~loc ~context:"parameter bound" lo;
       walk_expr_rule1 ~loc ~context:"parameter bound" hi
     | None -> ());
    let pname = match pd with PScalar s -> s.pname | PIndexed s -> s.pname in
    let pprior = match pd with PScalar s -> s.pprior | PIndexed s -> s.pprior in
    (match pprior with
     | Some ps ->
       List.iter (fun (_, e) ->
         walk_expr_rule1 ~loc
           ~context:(Printf.sprintf "prior on '%s'" pname) e
       ) ps.ps_args
     | None -> ())
  ) ctx.param_decls;

  (* ── let bindings: walk their body so a let used in a non-Add
       context still surfaces an in-body calendar+instant. The
       laundered case `let d = 6 'months; date(...) + d` is caught
       at the use site via `classify`'s let-table lookup; this walk
       catches `let bad = date("2020-02-24") + 6 'months` itself. ── *)
  List.iter (fun (lb : let_binding) ->
    walk_expr_rule1 ~loc:Diagnostics.no_loc
      ~context:(Printf.sprintf "let binding '%s'" lb.lname) lb.lbody
  ) ctx.let_bindings;

  (* ── init values ────────────────────────────────────────────────────── *)
  List.iter (fun (ie : init_entry) ->
    walk_expr_rule1 ~loc:(diag_loc_of_ast_ctx ctx ie.iloc)
      ~context:(Printf.sprintf "init '%s'" ie.icomp) ie.ivalue
  ) ctx.init_entries;

  (* ── simulate block: Rule 1 walk + bare-numeric W324 ────────────────── *)
  (match ctx.simulate with
   | Some sd ->
     walk_expr_rule1 ~loc:Diagnostics.no_loc ~context:"simulate.from" sd.sim_from;
     walk_expr_rule1 ~loc:Diagnostics.no_loc ~context:"simulate.to"   sd.sim_to;
     (* dt is a duration (step length), so a bare numeric dt is correct —
        no W324 here, unlike from/to which want calendar dates under origin. *)
     Option.iter
       (walk_expr_rule1 ~loc:Diagnostics.no_loc ~context:"simulate.dt")
       sd.sim_dt;
     warn_bare_numeric ~loc:Diagnostics.no_loc ~code:"W324" ~field:"from"
       ~hint:Time_typing.hint_bare_numeric_simulate sd.sim_from;
     warn_bare_numeric ~loc:Diagnostics.no_loc ~code:"W324" ~field:"to"
       ~hint:Time_typing.hint_bare_numeric_simulate sd.sim_to
   | None -> ());

  (* ── interventions and events: at-schedules + recurring cadences ────── *)
  let check_iv_list (ivs : intervention_decl list) ~label =
    List.iter (fun (iv : intervention_decl) ->
      let loc = diag_loc_of_ast_ctx ctx iv.ivloc in
      (* Action expressions can carry exprs too — fraction/count, set, add *)
      List.iter (fun a -> match a with
       | ATransfer kws -> List.iter (fun (_, e) ->
           walk_expr_rule1 ~loc ~context:label e) kws
       | ASet (_, _, e) -> walk_expr_rule1 ~loc ~context:label e
       | AAdd (_, _, e) -> walk_expr_rule1 ~loc ~context:label e) iv.ivaction;
      (* Schedule expressions: at-list, recurring cadences *)
      (match iv.ivschedule with
       | SAtTimes exprs ->
         List.iter (fun e ->
           walk_expr_rule1 ~loc ~context:(label ^ " at-time") e;
           warn_bare_numeric ~loc ~code:"W325" ~field:(label ^ " at[..]")
             ~hint:Time_typing.hint_bare_numeric_at_schedule e
         ) exprs
       | SRecurring (every, from_opt, until_opt) ->
         walk_expr_rule1 ~loc ~context:(label ^ ".every") every;
         check_recurring_cadence ~loc ~field:"every" every;
         (match from_opt with
          | Some e ->
            walk_expr_rule1 ~loc ~context:(label ^ ".from") e;
            check_recurring_cadence ~loc ~field:"from" e;
            warn_bare_numeric ~loc ~code:"W325" ~field:(label ^ ".from")
              ~hint:Time_typing.hint_bare_numeric_at_schedule e
          | None -> ());
         (match until_opt with
          | Some e ->
            walk_expr_rule1 ~loc ~context:(label ^ ".until") e;
            check_recurring_cadence ~loc ~field:"until" e;
            warn_bare_numeric ~loc ~code:"W325" ~field:(label ^ ".until")
              ~hint:Time_typing.hint_bare_numeric_at_schedule e
          | None -> ())
       | SEveryAtDay (period, day) ->
         walk_expr_rule1 ~loc ~context:(label ^ ".every") period;
         walk_expr_rule1 ~loc ~context:(label ^ ".at_day") day;
         check_recurring_cadence ~loc ~field:"every" period)
    ) ivs
  in
  check_iv_list ctx.interv_decls ~label:"intervention";
  check_iv_list ctx.event_decls  ~label:"event";

  (* ── observations: schedule expressions ─────────────────────────────── *)
  List.iter (fun (od : obs_decl) ->
    let loc = diag_loc_of_ast_ctx ctx od.oloc in
    (match od.oschedule with
     | Some (SchedEvery e) ->
       walk_expr_rule1 ~loc ~context:("observation '" ^ od.oname ^ "'.every") e;
       check_recurring_cadence ~loc ~field:"every" e
     | Some (SchedAt ts) ->
       List.iter (fun e ->
         walk_expr_rule1 ~loc ~context:("observation '" ^ od.oname ^ "'.at") e
       ) ts
     | None -> ())
  ) ctx.obs_decls;

  (* ── forcing functions: kwarg exprs (period, step, on=[...]) ──────────
     Note: bare-numeric `on=[...]` (Rule 4) is enforced at the
     existing periodic expansion site in `expand_time_function_one`,
     because that's where the EList is unwrapped. Here we still
     Rule-1-walk every kwarg in case a date() + 'months sneaks in. *)
  List.iter (fun (fd : func_decl) ->
    List.iter (fun (_, e) ->
      walk_expr_rule1 ~loc:Diagnostics.no_loc
        ~context:(Printf.sprintf "forcing '%s'" fd.fname) e
    ) fd.fargs
  ) ctx.func_decls;

  (* ── balance, output ────────────────────────────────────────────────── *)
  (match ctx.balance_decl with
   | Some bd ->
     walk_expr_rule1 ~loc:Diagnostics.no_loc
       ~context:(Printf.sprintf "balance '%s'" bd.bcomp) bd.bexpr
   | None -> ());

  ()

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

  (* gh#115 / 2026-05-26 upstream OCaml-compiler review Critical #5:
     scenario field names (enable / disable / compose / set / scale)
     were never validated against declared interventions / scenarios /
     parameters. A typo silently disabled nothing and ran as baseline
     — a direct wrong counterfactual when the scenario is selected at
     run time. Build the validation sets once and check every scenario
     before the IR build below. *)
  let intervention_names : (string, unit) Hashtbl.t =
    (* Accept both the family name ("sia") and every fully-expanded
       per-instance name ("sia_kano", "sia_lagos"). The runtime filter
       `resolve_enable_list` (rust/crates/cli/src/util.rs) accepts both an
       exact instance name and a family base_name, so a scenario that
       enables/disables a single expanded instance is legal — gh#130. The
       instance names mirror `expand_scheduled_actions` exactly: same
       cartesian product, same guard, same `ivname ^ "_" ^ parts`. *)
    let t = Hashtbl.create (List.length ctx.interv_decls) in
    List.iter (fun (iv : intervention_decl) ->
      Hashtbl.replace t iv.ivname ();
      let combos = cartesian_product iv.ivindices ctx in
      List.iter (fun env ->
        let pass_guard = match iv.ivguard with
          | None   -> true
          | Some g -> eval_guard ctx env g
        in
        if pass_guard then begin
          let parts = name_parts_from_bindings iv.ivindices env in
          let iv_name =
            if parts = [] then iv.ivname
            else iv.ivname ^ "_" ^ String.concat "_" parts
          in
          Hashtbl.replace t iv_name ()
        end
      ) combos
    ) ctx.interv_decls;
    t
  in
  let scenario_names : (string, unit) Hashtbl.t =
    let t = Hashtbl.create (List.length ctx.scenario_decls) in
    List.iter (fun (sd : scenario_decl) ->
      Hashtbl.replace t sd.scname ()
    ) ctx.scenario_decls;
    t
  in
  let parameter_names : (string, unit) Hashtbl.t =
    (* Accept both the family name ("N") and the fully-expanded
       per-stratum name ("N_rural", "N_urban"). The expansion table
       is populated by `build_lookup_tables` upstream of this call
       site (verified at line ~741: `ctx.expanded_param_tbl <- ept`).
       Multi-dim indexed params: `build_lookup_tables` populates the
       expanded set only for single-dim indexed params today; a
       multi-dim parameter's family name is still accepted via the
       PIndexed branch below. That mirrors today's user-facing
       behaviour and avoids false-positives on existing models. *)
    let t = Hashtbl.create (List.length ctx.param_decls) in
    List.iter (fun (pd : param_decl) ->
      let n = match pd with
        | PScalar  { pname; _ } -> pname
        | PIndexed { pname; _ } -> pname
      in
      Hashtbl.replace t n ()
    ) ctx.param_decls;
    Hashtbl.iter (fun k () -> Hashtbl.replace t k ()) ctx.expanded_param_tbl;
    t
  in
  let table_names tbl =
    Hashtbl.fold (fun k () acc -> k :: acc) tbl [] |> List.sort compare
  in
  let report_unknown ~code ~field ~scope ~scenario_name name names_tbl =
    Diagnostics.error ctx.diags
      ~code
      ~loc:Diagnostics.no_loc
      ~message:(Printf.sprintf
        "scenario '%s': %s names unknown %s '%s'"
        scenario_name field scope name)
      ~hint:(Printf.sprintf
        "declare %s '%s' first, or fix the typo. Available %ss: %s"
        scope name scope
        (let ns = table_names names_tbl in
         if ns = [] then "(none declared)" else String.concat ", " ns))
      ()
  in

  (* Pass 2: validate every scenario's field references.
     Performed on raw `own` (not after parent merge) so each error
     names the scenario that authored the bad reference, not its
     descendant — clearer for diagnostic-driven fixes. *)
  List.iter (fun sd ->
    let own = collect_own_fields sd in
    let name = sd.scname in
    (* `fitted` is reserved: it labels the no-overlay row (the fitted model,
       no scenario applied) in the `scenario` column emitted by `camdl fit
       predict`. A preset by that name would shadow the reserved value and make
       rows ambiguous. Reject it with a migration-style diagnostic that names the
       reservation and the fix (rename the scenario). *)
    if name = "fitted" then
      Diagnostics.error ctx.diags
        ~code:"E291"
        ~loc:Diagnostics.no_loc
        ~message:(Printf.sprintf
          "scenario name '%s' is reserved" name)
        ~hint:"`fitted` labels the no-overlay row (the fitted model, no \
               scenario applied) in the `scenario` column of `camdl fit \
               predict` output. Rename the scenario so it does not collide \
               with the reserved value."
        ();
    List.iter (fun n ->
      if not (Hashtbl.mem intervention_names n) then
        report_unknown ~code:"E267" ~field:"enable"
          ~scope:"intervention" ~scenario_name:name n intervention_names
    ) own.rs_enable;
    List.iter (fun n ->
      if not (Hashtbl.mem intervention_names n) then
        report_unknown ~code:"E267" ~field:"disable"
          ~scope:"intervention" ~scenario_name:name n intervention_names
    ) own.rs_disable;
    List.iter (fun n ->
      if not (Hashtbl.mem scenario_names n) then
        report_unknown ~code:"E269" ~field:"compose"
          ~scope:"scenario" ~scenario_name:name n scenario_names
    ) own.rs_compose;
    List.iter (fun (k, _) ->
      if not (Hashtbl.mem parameter_names k) then
        report_unknown ~code:"E268" ~field:"set"
          ~scope:"parameter" ~scenario_name:name k parameter_names
    ) own.rs_set;
    List.iter (fun (k, _) ->
      if not (Hashtbl.mem parameter_names k) then
        report_unknown ~code:"E268" ~field:"scale"
          ~scope:"parameter" ~scenario_name:name k parameter_names
    ) own.rs_scale
  ) ctx.scenario_decls;

  (* Pass 3: for each scenario, resolve parents then emit IR preset. *)
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
  |> List.find_opt (fun td -> is_expansion_of ~base:td.trname ename)
  |> Option.map (fun td -> td.trname)

(** Same invariant: compartment expanded names are {base}_{dim_values}.
    Same longest-prefix-wins fix as find_base_trname above. *)
let find_base_compname ctx expanded_name =
  List.sort (fun a b ->
    compare (String.length b.cname) (String.length a.cname)) ctx.comp_decls
  |> List.find_opt (fun cd -> is_expansion_of ~base:cd.cname expanded_name)
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
    | Ir.Const _ | Ir.Param _ | Ir.Time | Ir.Dt | Ir.Projected
    | Ir.ObsColumnRef _ | Ir.TimeFunc _ -> acc
    | Ir.UncheckedDim u -> collect_numerator_pops acc u.inner
    (* Every term of a sum is a numerator contribution (like Add). *)
    | Ir.Reduce terms -> List.fold_left collect_numerator_pops acc terms
    | Ir.BindingRef _ -> acc
    | Ir.PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"
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
  | Ir.Reduce terms -> List.exists expr_contains_param_or_pop terms
  | Ir.BindingRef _ -> false
  | Ir.PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"
  | Ir.Const _ | Ir.Time | Ir.Dt | Ir.Projected | Ir.ObsColumnRef _ | Ir.TimeFunc _ -> false

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
  | Ir.Reduce terms -> List.iter (walk_expr_for_l401 ~on_match) terms
  | Ir.BindingRef _ -> ()
  | Ir.PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"
  | Ir.Const _ | Ir.Param _ | Ir.Pop _ | Ir.PopSum _
  | Ir.Time | Ir.Dt | Ir.Projected | Ir.ObsColumnRef _ | Ir.TimeFunc _ -> ()

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

(* ── L403 lint: manual re-conversion of an already-rescaled rate forcing ──
   (gh#13)

   A forcing declared with a per-time tier-3 unit literal (`'per_day`,
   `'per_week`, `'per_month`, `'per_year`) has its stored values rescaled to
   the model `time_unit` at expand time (see `scale_expr` / `unit_to_model_time`
   above and spec §7 "Required unit literal"). So `birthrate(t)` already
   returns a value in the model time unit — e.g. a `'per_year` forcing under
   `time_unit = 'days` is a PER-DAY value at every reference site.

   A user who reads the `'per_year` annotation as a passive type tag and
   "converts" manually — `birthrate(t) * pop(t) / 365.25` — divides a SECOND
   time, producing a rate ~365× too small. The dim-checker cannot catch it:
   dividing a rate (T⁻¹) by a bare dimensionless constant preserves the
   dimension, and the dim system tracks (P_exp, T_exp) but not the time-SCALE,
   so per-year and per-day are the same dimension. The rescale is silent at
   every reference site — there is no signal at the use site. This lint is that
   signal (make-loud; the real fix, a scale-aware dim system, is out of scope).

   The lint fires ONLY for a forcing that was ACTUALLY rescaled at load: its
   declared dim is a rate (T⁻¹) AND its unit differs from the model `time_unit`,
   so its rescale factor `s = unit_to_model_time ctx 1.0 funit ≠ 1.0`. A
   same-unit rate forcing — e.g. a `'per_day` forcing under `time_unit = 'days`,
   where `s = 1` and NO conversion happened at load — is left alone: dividing it
   by a constant is not the double-conversion bug. The matched divisor is checked
   against THIS forcing's OWN double-convert magnitude `m = 1/s`, not a generic
   set of calendar constants — so a structural divisor that merely collides with
   a calendar constant (`import_rate('per_year)(t) / 12`, where `/12` is 12
   provinces, not months) does NOT fire.

   Shape matched (post-expansion, per transition rate / hoisted binding body):
     - `Div` whose DENOMINATOR is a BARE `Const` c (NOT a unit literal —
       `UncheckedDim` wraps `1 'years` / `365 'days`, and writing the divisor
       as a unit literal is exactly the recommended fix, so it must NOT fire)
       with c ≈ m for the rate forcing referenced in the NUMERATOR subtree;
     - `Mul` by the RECIPROCAL of m — a bare `Const c` with c ≈ 1/m = s, or an
       unfolded `Const 1 / Const m` — where the OTHER operand references that
       rate forcing. (The lint runs inside `expand_detail`, BEFORE constant
       folding, so `1/365.25` is still `Div{Const 1, Const 365.25}`, not a single
       folded `Const`.)

   The magnitude `m` is the forcing's own `1/s` (a `'per_year` forcing under
   `time_unit = 'days` has `s = 1/365.2425`, so `m ≈ 365.2425`), matched within a
   0.5% relative band. A genuine residual ambiguity remains for a `'per_week`
   forcing ÷ 7 or a `'per_month` forcing ÷ 30.44, which is indistinguishable from
   a real double-convert — accepted.

   Warning, not a hard error, for 0.2.0: the magnitude match is a heuristic, and
   a hard error needs the per-site suppression escape hatch (gh#55
   `#[allow(...)]` / gh#56 `--allow=`) which does not exist yet. Promote to an
   error once that lands. See docs/dev/warning-catalog.md §L403. *)

(* Bare numeric constant ONLY. A unit-annotated divisor (`1 'years`,
   `365 'days`) wraps in `UncheckedDim` and IS the recommended fix, so it must
   not match. (`'ratio` / `'count` literals lower to a bare `Const` and are
   correctly indistinguishable from a bare number here.) Distinct from L401's
   `as_const`, which deliberately also unwraps `UncheckedDim`. *)
let as_bare_const = function
  | Ir.Const c -> Some c
  | _ -> None

(* 0.5% relative band — conservative, keeps false positives near zero. *)
let l403_tol = 0.005

(* c ≈ [expected] within the relative band (the `Div`-denominator case).
   [expected] is the referenced forcing's OWN double-convert magnitude m = 1/s,
   NOT a member of a generic conversion-constant set. *)
let l403_div_magnitude_match c expected =
  c > 0.0 && expected > 0.0 && Float.abs (c -. expected) <= l403_tol *. expected

(* A multiplicative factor equal to the RECIPROCAL of the forcing's magnitude
   [expected] (m = 1/s), i.e. the forcing's own rescale factor s. Two forms:
     - bare `Const c` with c ≈ 1/expected       (folded / hand-written reciprocal);
     - `Const a / Const m`, a ≈ 1, m ≈ expected  (unfolded `1/365.25`, since the
       lint runs pre-constant-fold). *)
let l403_reciprocal_match factor expected =
  match factor with
  | Ir.Const c when c > 0.0 && expected > 0.0 ->
    Float.abs (c -. (1.0 /. expected)) <= l403_tol *. (1.0 /. expected)
  | Ir.BinOp { op = Ir.Div; left = Ir.Const a; right = Ir.Const m } ->
    Float.abs (a -. 1.0) <= l403_tol && l403_div_magnitude_match m expected
  | _ -> false

(* First rate-dimensioned forcing name referenced anywhere in [e], if any.
   [is_rate_forcing] tests membership in the set of forcings whose declared
   dim is a rate (0, -1). A `BindingRef` numerator (a separately-hoisted `let`
   holding the forcing ref) is a known heuristic gap — not resolved here. *)
let rec l403_rate_forcing_in is_rate_forcing = function
  | Ir.TimeFunc name when is_rate_forcing name -> Some name
  | Ir.BinOp { left; right; _ } ->
    (match l403_rate_forcing_in is_rate_forcing left with
     | Some _ as r -> r
     | None -> l403_rate_forcing_in is_rate_forcing right)
  | Ir.UnOp { arg; _ } -> l403_rate_forcing_in is_rate_forcing arg
  | Ir.Cond { pred; then_; else_ } ->
    (match l403_rate_forcing_in is_rate_forcing pred with
     | Some _ as r -> r
     | None ->
       (match l403_rate_forcing_in is_rate_forcing then_ with
        | Some _ as r -> r
        | None -> l403_rate_forcing_in is_rate_forcing else_))
  | Ir.UncheckedDim u -> l403_rate_forcing_in is_rate_forcing u.inner
  | Ir.TableLookup (_, args) ->
    List.find_map (l403_rate_forcing_in is_rate_forcing) args
  | Ir.Reduce terms -> List.find_map (l403_rate_forcing_in is_rate_forcing) terms
  | Ir.TimeFunc _ | Ir.Const _ | Ir.Param _ | Ir.Pop _ | Ir.PopSum _
  | Ir.Time | Ir.Dt | Ir.Projected | Ir.ObsColumnRef _ | Ir.BindingRef _ -> None
  | Ir.PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"

(* Detect the L403 shape rooted at [e]. [mags] maps each already-rescaled rate
   forcing (dim (0,-1), rescale factor s ≠ 1) to its OWN double-convert magnitude
   m = 1/s; a forcing absent from [mags] is not a candidate (a non-rate forcing,
   or a same-unit rate forcing that was never rescaled). Returns
   [Some (forcing_name, magnitude)] for the diagnostic. *)
let detect_l403_at_node mags e =
  let is_rate_forcing name = Hashtbl.mem mags name in
  match e with
  | Ir.BinOp { op = Ir.Div; left = num; right = denom } ->
    (match as_bare_const denom with
     | Some c ->
       (match l403_rate_forcing_in is_rate_forcing num with
        | Some fname ->
          let expected = Hashtbl.find mags fname in
          if l403_div_magnitude_match c expected then Some (fname, c) else None
        | None -> None)
     | None -> None)
  | Ir.BinOp { op = Ir.Mul; left; right } ->
    let check factor other =
      match l403_rate_forcing_in is_rate_forcing other with
      | Some fname ->
        let expected = Hashtbl.find mags fname in
        if l403_reciprocal_match factor expected then Some (fname, expected)
        else None
      | None -> None
    in
    (match check left right with Some _ as r -> r | None -> check right left)
  | _ -> None

let rec walk_expr_for_l403 mags ~on_match e =
  (match detect_l403_at_node mags e with
   | Some (fname, mag) -> on_match fname mag
   | None -> ());
  match e with
  | Ir.BinOp { left; right; _ } ->
    walk_expr_for_l403 mags ~on_match left;
    walk_expr_for_l403 mags ~on_match right
  | Ir.UnOp { arg; _ } -> walk_expr_for_l403 mags ~on_match arg
  | Ir.Cond { pred; then_; else_ } ->
    walk_expr_for_l403 mags ~on_match pred;
    walk_expr_for_l403 mags ~on_match then_;
    walk_expr_for_l403 mags ~on_match else_
  | Ir.UncheckedDim u -> walk_expr_for_l403 mags ~on_match u.inner
  | Ir.TableLookup (_, args) ->
    List.iter (walk_expr_for_l403 mags ~on_match) args
  | Ir.Reduce terms ->
    List.iter (walk_expr_for_l403 mags ~on_match) terms
  | Ir.Const _ | Ir.Param _ | Ir.Pop _ | Ir.PopSum _
  | Ir.Time | Ir.Dt | Ir.Projected | Ir.ObsColumnRef _
  | Ir.TimeFunc _ | Ir.BindingRef _ -> ()
  | Ir.PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"

let lint_l403 ctx (transitions : Ir.transition list)
    (bindings : Ir.binding list) (time_functions : Ir.time_function list)
    (forcing_scales : (string * float) list) =
  (* Candidate forcings, mapped to their OWN double-convert magnitude m = 1/s.
     A forcing qualifies only when BOTH hold:
       1. its declared dimension is a rate (T⁻¹) — a `'ratio` / `'count` /
          duration forcing is not a rate, so dividing it is not this bug; AND
       2. its load-time rescale factor s ≠ 1.0 — i.e. its unit differs from the
          model `time_unit`, so a conversion ACTUALLY happened when its stored
          values were baked. A same-unit rate forcing (`'per_day` under
          `time_unit = 'days`) has s = 1.0 (round-trip exact for every current
          unit) and is excluded: there was no first conversion to double. *)
  let mags = Hashtbl.create 8 in
  List.iter (fun (tf : Ir.time_function) ->
    match tf.Ir.dim with
    | (0, -1) ->
      (match List.assoc_opt tf.Ir.name forcing_scales with
       | Some scale when scale <> 1.0 && scale > 0.0 ->
         Hashtbl.replace mags tf.Ir.name (1.0 /. scale)
       | _ -> ())
    | _ -> ()
  ) time_functions;
  if Hashtbl.length mags = 0 then ()   (* nothing to flag — skip the walk *)
  else begin
    (* Fire at most once per (site, forcing): a forcing divided twice in one
       expression must not double-emit, while two distinct bad forcings in the
       same site each surface. *)
    let seen = Hashtbl.create 8 in
    let emit ~site fname mag =
      let key = site ^ "|" ^ fname in
      if not (Hashtbl.mem seen key) then begin
        Hashtbl.replace seen key ();
        Diagnostics.warning ctx.diags ~code:"L403" ~loc:Diagnostics.no_loc
          ~message:(Printf.sprintf
            "%s divides the rate forcing '%s'(t) by the time-conversion constant \
             %g — but '%s' is declared as a per-time (rate) forcing and is \
             ALREADY rescaled to the model time_unit at load, so '%s'(t) returns \
             the model-time value; dividing by %g here applies the conversion a \
             SECOND time (rate ~%g× too small)"
            site fname mag fname fname mag mag)
          ~hint:(Printf.sprintf
            "drop the manual conversion — '%s'(t) is already in the model \
             time_unit, so use it directly. If you genuinely need a further \
             scale, write the factor as a dimensioned unit literal (not a bare \
             number) so the dim-checker can validate it. See \
             docs/dev/warning-catalog.md §L403 and spec §7 'Required unit literal'."
            fname)
          ()
      end
    in
    List.iter (fun (t : Ir.transition) ->
      walk_expr_for_l403 mags t.Ir.rate
        ~on_match:(emit ~site:(Printf.sprintf "transition '%s' rate" t.Ir.name))
    ) transitions;
    List.iter (fun (b : Ir.binding) ->
      walk_expr_for_l403 mags b.Ir.bexpr
        ~on_match:(emit ~site:(Printf.sprintf "binding '%s'" b.Ir.bname))
    ) bindings
  end

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
  (* Staged-residence (`via`) lowering — an AST→AST pre-pass that rewrites every
     `via erlang(...)` transition into the manual stratified-`consecutive` form
     (stage the source, emit the chain + exit, redirect inflow/init, scale the
     rate). Runs BEFORE resolve_dimensions / check_declaration_names so the
     synthesized stage dimension, stratify entry, and chain transitions go
     through the ordinary stratification + consecutive machinery unchanged. After
     this pass no `via` transition survives (the E243 placeholder in
     [expand_transitions_counted] is unreachable). *)
  lower_via_transitions ctx;
  (* M14 (gh#98): validate origin date up front, before origin_rata_die /
     date() conversion derives values from it. *)
  check_origin ctx;
  (* Pass 1: resolve dimensions {} block, build dim_registry *)
  resolve_dimensions ctx;
  (* gh#117: reject duplicate-within-namespace and cross-namespace
     declaration names (both base AND fully-expanded/stratified names),
     naming both declarations. Runs after resolve_dimensions (so expanded
     names are known) and before build_lookup_tables (so the silent
     Hashtbl.replace last-wins never gets a chance to mask a collision). *)
  check_declaration_names ctx;
  (* Build O(1) lookup tables for resolve_expr *)
  build_lookup_tables ctx;
  (* W103 shadowing check: let bindings vs stratum values *)
  check_shadowing ctx;
  (* E283: a sum/binder var must not shadow an enclosing index/bound var *)
  check_no_shadowing ctx;
  (* W105: warn on the per-(p,q) coupling antipattern (O(P^2) transitions) *)
  check_quadratic_coupling ctx;
  (* E236: hierarchical-prior cycle / self-reference detection (#3 gate 2) *)
  check_hierarchical_cycles ctx;
  (* E217: check that guard expressions only reference dim levels / loop vars *)
  check_guards ctx;
  (* Phase 1 of the 2026-05-22 typed-time proposal:
     Surface-level time-typing rules (Rule 1: Instant + CalendarDuration;
     Rule 2: time_unit + origin; Rule 5: bare-numeric in time positions;
     Rule 7: calendar cadences in recurring schedules). Runs before
     resolve_expr drops unit-literal provenance from the AST. *)
  check_surface_time_typing ctx;
  (* Save original transitions before desugaring *)
  ctx.orig_transitions <- ctx.transitions;
  (* Resolve tables BEFORE transition expansion: tables are compile-time
     constants (they depend only on dimensions, not transitions), and a
     restricted-sum `where` predicate needs their values during resolve_expr.
     Indexed into ctx.table_index for the predicate; reused for Ir.tables. *)
  let resolved_tables = expand_tables ctx in
  build_table_index ctx resolved_tables;
  let expanded_comps = expand_compartments ctx in
  let (expanded_trs, filtered_n) = expand_transitions_counted ctx in
  lint_l401 ctx expanded_trs;
  let ms = build_model_structure ctx expanded_trs in
  (* Expand forcings once, capturing the per-forcing load-time rescale factors
     [forcing_scales] so [lint_l403] can distinguish an actually-rescaled forcing
     (scale ≠ 1) from a same-unit one (scale = 1, no conversion happened). *)
  let (expanded_time_functions, forcing_scales) = expand_time_functions ctx in
  let model = {
    Ir.name               = name;
    Ir.version            = "0.3";
    Ir.time_unit          = unit_lit_to_string ctx.time_unit;
    Ir.description        = ctx.description;
    Ir.origin             = ctx.origin;
    (* Pre-resolve the origin to its proleptic-Gregorian day number so the
       runtime never re-parses the origin string (2026-05-22 §6.2). The
       integer is derived (never hand-edited) and uses the same `days_of_date`
       the date() literal path uses, so it cannot drift from caltime. *)
    Ir.origin_rata_die    =
      (match ctx.origin with
       | None -> None
       | Some s ->
         (* The origin was range-validated up-front by `check_origin` (M14,
            gh#98); on the clean path this always parses. If it didn't, the
            E223 already fired — fall back to None rather than emit garbage. *)
         (match parse_iso_date s with
          | Ok (y, m, d) -> Some (days_of_date y m d)
          | Error _ -> None));
    Ir.compartments       = expanded_comps;
    Ir.transitions        = expanded_trs;
    Ir.ode_equations      = expand_ode_equations ctx;
    Ir.time_functions     = expanded_time_functions;
    Ir.tables             = resolved_tables;
    Ir.interventions      = expand_interventions ctx;
    Ir.observations       = expand_observations ctx;
    Ir.parameters         = expand_parameters ctx;
    Ir.bindings           = [];   (* filled below from ctx.hoisted_rev once all resolution is done *)
    Ir.per_eval_bindings  = [];   (* gh#272 LICM: empty until the LICM pass runs (post-autodiff) *)
    Ir.initial_conditions = expand_init ctx;
    Ir.ic_grad            = [];
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
    Ir.doc_index = build_doc_index ctx;
    Ir.quantities         = expand_quantities ctx;
    Ir.contrasts          = expand_contrasts ctx;
  } in
  (* Fix B: the record above is fully forced here, so every resolve_expr call
     (transitions, ode, observations, balance) has run and ctx.hoisted_rev
     holds all extracted bindings in reverse-topological order. *)
  let model = { model with Ir.bindings = collect_hoisted_bindings ctx } in
  (* L403 (gh#13): flag a manual re-conversion of an already-rescaled per-time
     forcing (`birthrate(t) / 365.25`). Runs after bindings are collected so it
     can also walk hoisted-`let` bodies, where the `forcing * pop / const` idiom
     often lands. *)
  lint_l403 ctx model.Ir.transitions model.Ir.bindings
    model.Ir.time_functions forcing_scales;
  (* gh#204: reject reactive triggers that read an undeclared observation stream
     (post-pass: both interventions and observations are now expanded). *)
  validate_reactive_streams ctx model;
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
