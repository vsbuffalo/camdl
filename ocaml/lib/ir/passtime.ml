(** Per-stage processor time for one compile, written to standard error when
    the environment variable [CAMDL_TIME_PASSES] is set.

    The driver wraps each stage of a compile in {!time} (or measures it itself
    and calls {!record}), under a short label: [parse], [expand], [validate],
    [dimcheck], [lint], [autodiff], [declared-names], the differentiation of
    observations, projections and initial conditions ([autodiff-obs] and
    siblings), [constant_fold], [licm] and, in [camdlc], [serialize]. At the end
    of a compile [camdlc] calls {!dump}, which prints one row per recorded
    stage, in the order recorded, with its seconds and its share of the sum,
    then a [TOTAL] row. The total is the sum of the recorded stages, not the
    elapsed time of the process.

    The variable is read once per process: any value other than empty or [0]
    turns timing on. When it is off, {!time} only calls its function and
    {!record} and {!dump} do nothing, so a normal compile pays nothing and its
    output is unchanged; when it is on, the report goes to standard error and
    the IR on standard output is still byte-for-byte the same.

    Times are [Sys.time], the processor time of the process, which includes
    garbage collection. For the single-threaded compiler this is close to
    elapsed time, and it keeps the [unix] library out of the [ir] and
    [compiler] libraries. Records accumulate for the life of the process and
    {!dump} does not clear them, so a process that compiles several models
    reports their sum. *)

let enabled : bool Lazy.t =
  lazy (match Sys.getenv_opt "CAMDL_TIME_PASSES" with
        | Some s -> s <> "" && s <> "0"
        | None -> false)

(* The recorded (label, seconds) pairs, newest first. *)
let records : (string * float) list ref = ref []

(** [record label dt] logs [dt] seconds against [label]. No-op when disabled. *)
let record (label : string) (dt : float) : unit =
  if Lazy.force enabled then records := (label, dt) :: !records

(** [time label f] runs [f], records its processor time under [label], and
    returns its result. Exceptions propagate (no time recorded on failure). *)
let time (label : string) (f : unit -> 'a) : 'a =
  if not (Lazy.force enabled) then f ()
  else begin
    let t0 = Sys.time () in
    let r = f () in
    record label (Sys.time () -. t0);
    r
  end

(** [dump ()] writes the collected breakdown to stderr. No-op when disabled or
    when nothing was recorded. Safe to call once at the end of a compile. *)
let dump () : unit =
  if Lazy.force enabled && !records <> [] then begin
    let rows = List.rev !records in
    let total = List.fold_left (fun a (_, dt) -> a +. dt) 0.0 rows in
    Printf.eprintf "\n[camdl pass timing]  (processor seconds, CAMDL_TIME_PASSES)\n";
    List.iter (fun (label, dt) ->
      let pct = if total > 0.0 then 100.0 *. dt /. total else 0.0 in
      Printf.eprintf "  %-12s %9.3f s  %5.1f%%\n" label dt pct
    ) rows;
    Printf.eprintf "  %-12s %9.3f s  100.0%%\n" "TOTAL" total
  end
