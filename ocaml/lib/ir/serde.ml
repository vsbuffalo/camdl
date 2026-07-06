(* serde.ml — JSON serialization/deserialization for the compartmental IR.
 *
 * Hand-written rather than ppx-generated to maintain exact control over
 * the JSON wire format. The Rust backend (serde) deserializes this JSON,
 * so field names and structure must match ir/src/*.rs exactly. We avoid
 * ppx_deriving_yojson because:
 *
 * 1. ppx generates default field names from OCaml identifiers, which may
 *    differ from the Rust serde names. Every mismatch would need a
 *    [@key "..."] annotation — ~40 types × multiple fields = fragile.
 *
 * 2. The JSON shape is the IR contract between OCaml and Rust. Hand-written
 *    serde makes the contract explicit and auditable. ppx would hide it
 *    behind generated code.
 *
 * 3. Some types need custom serialization (e.g., expr uses tagged unions
 *    with specific key names that ppx wouldn't generate naturally).
 *
 * When adding a new field to ir.mli:
 *   1. Add the field to the type in ir.mli and ir.ml
 *   2. Add _to_json and _of_json for it HERE (side by side)
 *   3. Add the corresponding Rust type in ir/src/
 *)

open Ir

(* ── Serialize helpers ───────────────────────────────────────────────────── *)

let opt_field name f = function
  | None   -> []
  | Some v -> [(name, f v)]

let str  s    : Yojson.Safe.t = `String s
let flt  f    : Yojson.Safe.t = `Float f
let bool b    : Yojson.Safe.t = `Bool b
let null      : Yojson.Safe.t = `Null
let obj  kvs  : Yojson.Safe.t = `Assoc kvs
let arr  xs   : Yojson.Safe.t = `List xs
let int  n    : Yojson.Safe.t = `Int n

(* ── Deserialize helpers ─────────────────────────────────────────────────── *)

exception DeserError of string

let fail fmt = Printf.ksprintf (fun s -> raise (DeserError s)) fmt

let member key = function
  | `Assoc kvs -> (
      match List.assoc_opt key kvs with
      | Some v -> v
      | None   -> fail "missing field '%s'" key
    )
  | _ -> fail "expected object, got non-object while looking for '%s'" key

let member_opt key = function
  | `Assoc kvs -> List.assoc_opt key kvs
  | _ -> fail "expected object while looking for optional '%s'" key

let as_string = function
  | `String s -> s
  | `Int n    -> string_of_int n
  | j -> fail "expected string, got %s" (Yojson.Safe.to_string j)

let as_float = function
  | `Float f -> f
  | `Int n   -> float_of_int n
  | j -> fail "expected number, got %s" (Yojson.Safe.to_string j)

let as_int = function
  | `Int n   -> n
  | `Float f ->
    let n = int_of_float f in
    if float_of_int n = f then n
    else fail "expected integer, got float %f" f
  | j -> fail "expected integer, got %s" (Yojson.Safe.to_string j)

let as_bool = function
  | `Bool b -> b
  | j -> fail "expected bool, got %s" (Yojson.Safe.to_string j)

let as_list = function
  | `List xs -> xs
  | j -> fail "expected array, got %s" (Yojson.Safe.to_string j)

let as_assoc = function
  | `Assoc kvs -> kvs
  | j -> fail "expected object, got %s" (Yojson.Safe.to_string j)

let opt_null f = function
  | `Null -> None
  | j     -> Some (f j)

(* ── Expression ──────────────────────────────────────────────────────────── *)

let bin_op_str = function
  | Add -> "add" | Sub -> "sub" | Mul -> "mul"
  | Div -> "div" | Pow -> "pow" | Mod -> "mod" | Min -> "min" | Max -> "max"
  | Eq  -> "eq"  | Neq -> "neq" | Lt  -> "lt"  | Gt  -> "gt"
  | Le  -> "le"  | Ge  -> "ge"

let un_op_str = function
  | Neg -> "neg" | Exp -> "exp" | Log -> "log"
  | Sqrt -> "sqrt" | Abs -> "abs" | Floor -> "floor" | Ceil -> "ceil"
  | Sin -> "sin" | Cos -> "cos" | Tanh -> "tanh"

let rec expr_to_json (e : expr) : Yojson.Safe.t =
  match e with
  | Const v      -> obj [("const", flt v)]
  | Param p      -> obj [("param", str p)]
  | Pop   p      -> obj [("pop",   str p)]
  | PopSum ps    -> obj [("pop_sum", arr (List.map str ps))]
  | Reduce terms -> obj [("reduce", arr (List.map expr_to_json terms))]
  | BindingRef n -> obj [("binding_ref", str n)]
  | PerEvalRef n -> obj [("per_eval_ref", str n)]
  | Time         -> obj [("time", null)]
  | Dt           -> obj [("dt", null)]
  | Projected    -> obj [("projected", null)]
  | ObsColumnRef c -> obj [("obs_column_ref", str c)]
  | BinOp b      ->
    obj [("bin_op", obj [
      ("op",    str (bin_op_str b.op));
      ("left",  expr_to_json b.left);
      ("right", expr_to_json b.right);
    ])]
  | UnOp u       ->
    obj [("un_op", obj [
      ("op",  str (un_op_str u.op));
      ("arg", expr_to_json u.arg);
    ])]
  | Cond c       ->
    obj [("cond", obj [
      ("pred", expr_to_json c.pred);
      ("then", expr_to_json c.then_);
      ("else", expr_to_json c.else_);
    ])]
  | TimeFunc n   ->
    obj [("time_func", obj [("name", str n)])]
  | TableLookup (tbl, idxs) ->
    obj [("table_lookup", obj [
      ("table",   str tbl);
      ("indices", arr (List.map expr_to_json idxs));
    ])]
  | UncheckedDim u ->
    obj [("unchecked_dim", obj [
      ("inner",  expr_to_json u.inner);
      ("dim",    arr [int u.dim_p; int u.dim_t]);
      ("reason", str u.reason);
    ])]

let bin_op_of_str = function
  | "add" -> Add | "sub" -> Sub | "mul" -> Mul
  | "div" -> Div | "pow" -> Pow | "mod" -> Mod | "min" -> Min | "max" -> Max
  | "eq"  -> Eq  | "neq" -> Neq | "lt"  -> Lt  | "gt"  -> Gt
  | "le"  -> Le  | "ge"  -> Ge
  | s -> fail "unknown bin_op '%s'" s

let un_op_of_str = function
  | "neg" -> Neg | "exp" -> Exp   | "log" -> Log
  | "sqrt" -> Sqrt | "abs" -> Abs | "floor" -> Floor | "ceil" -> Ceil
  | "sin" -> Sin | "cos" -> Cos | "tanh" -> Tanh
  | s -> fail "unknown un_op '%s'" s

let rec expr_of_json (j : Yojson.Safe.t) : expr =
  match j with
  | `Assoc kvs ->
    let keys = List.map fst kvs in
    (match keys with
    | ["const"]        -> Const (as_float (List.assoc "const" kvs))
    | ["param"]        -> Param (as_string (List.assoc "param" kvs))
    | ["pop"]          -> Pop   (as_string (List.assoc "pop" kvs))
    | ["pop_sum"]      -> PopSum (List.map as_string (as_list (List.assoc "pop_sum" kvs)))
    | ["reduce"]       -> Reduce (List.map expr_of_json (as_list (List.assoc "reduce" kvs)))
    | ["binding_ref"]  -> BindingRef (as_string (List.assoc "binding_ref" kvs))
    | ["per_eval_ref"] -> PerEvalRef (as_string (List.assoc "per_eval_ref" kvs))
    | ["time"]         -> Time
    | ["dt"]           -> Dt
    | ["projected"]    -> Projected
    | ["obs_column_ref"] -> ObsColumnRef (as_string (List.assoc "obs_column_ref" kvs))
    | ["bin_op"]       ->
      let b = List.assoc "bin_op" kvs in
      BinOp {
        op    = bin_op_of_str (as_string (member "op" b));
        left  = expr_of_json (member "left" b);
        right = expr_of_json (member "right" b);
      }
    | ["un_op"]        ->
      let u = List.assoc "un_op" kvs in
      UnOp {
        op  = un_op_of_str (as_string (member "op" u));
        arg = expr_of_json (member "arg" u);
      }
    | ["cond"]         ->
      let c = List.assoc "cond" kvs in
      Cond {
        pred  = expr_of_json (member "pred" c);
        then_ = expr_of_json (member "then" c);
        else_ = expr_of_json (member "else" c);
      }
    | ["time_func"]    ->
      let tf = List.assoc "time_func" kvs in
      TimeFunc (as_string (member "name" tf))
    | ["table_lookup"] ->
      let tl = List.assoc "table_lookup" kvs in
      let tbl  = as_string (member "table" tl) in
      let idxs = List.map expr_of_json (as_list (member "indices" tl)) in
      TableLookup (tbl, idxs)
    | ["unchecked_dim"] ->
      let u = List.assoc "unchecked_dim" kvs in
      let dim_arr = as_list (member "dim" u) in
      (match dim_arr with
       | [p; t] ->
         UncheckedDim {
           inner  = expr_of_json (member "inner" u);
           dim_p  = as_int p;
           dim_t  = as_int t;
           reason = as_string (member "reason" u);
         }
       | _ -> fail "unchecked_dim.dim must be [P, T] pair")
    | _ ->
      fail "unrecognised expr object with keys [%s]" (String.concat ", " keys)
    )
  | _ -> fail "expr must be a JSON object, got %s" (Yojson.Safe.to_string j)

(* ── Compartment ─────────────────────────────────────────────────────────── *)

let compartment_kind_to_json = function
  | Integer -> str "integer"
  | Real    -> str "real"

let compartment_kind_of_json j =
  match as_string j with
  | "integer" -> Integer
  | "real"    -> Real
  | s -> fail "unknown compartment kind '%s'" s

let compartment_to_json (c : compartment) : Yojson.Safe.t =
  obj [("name", str c.name); ("kind", compartment_kind_to_json c.kind)]

let compartment_of_json j : compartment =
  { name = as_string (member "name" j);
    kind = compartment_kind_of_json (member "kind" j) }

(* ── Transition ──────────────────────────────────────────────────────────── *)

let stoich_entry_to_json ((name, delta) : stoichiometry_entry) : Yojson.Safe.t =
  arr [str name; int delta]

let stoich_entry_of_json j =
  match as_list j with
  | [name; delta] -> (as_string name, as_int delta)
  | _ -> fail "stoichiometry entry must be a 2-element array"

let metadata_to_json (m : transition_metadata) : Yojson.Safe.t =
  obj [
    ("origin_kind",        match m.origin_kind with None -> null | Some s -> str s);
    ("source_compartment", match m.source_compartment with None -> null | Some s -> str s);
    ("dest_compartment",   match m.dest_compartment   with None -> null | Some s -> str s);
  ]

let metadata_of_json j =
  { origin_kind        = opt_null as_string (match member_opt "origin_kind"        j with Some v -> v | None -> `Null);
    source_compartment = opt_null as_string (match member_opt "source_compartment" j with Some v -> v | None -> `Null);
    dest_compartment   = opt_null as_string (match member_opt "dest_compartment"   j with Some v -> v | None -> `Null);
  }

(* ── Derivative entries (obs/σ² gradient surface) ──────────────────────────
   Externally-tagged single-key objects, mirroring `ir::deriv::DerivEntry`'s
   Rust `#[serde(rename_all="snake_case")]`:
     DEGrad e        -> {"grad": <expr>}
     DEUnsupported   -> {"unsupported": {"node": "…", "code": "<reason>"}}
   A grad_map serialises as a plain {param → deriv_entry} object, exactly like
   the transition rate_grad's {param → expr}. *)
let deriv_entry_to_json (de : deriv_entry) : Yojson.Safe.t =
  match de with
  | DEGrad e -> obj [("grad", expr_to_json e)]
  | DEUnsupported { node; code } ->
    obj [("unsupported",
          obj [("node", str node); ("code", str (unsupported_reason_name code))])]

let deriv_entry_of_json j : deriv_entry =
  match j with
  | `Assoc [(key, v)] -> (
    match key with
    | "grad" -> DEGrad (expr_of_json v)
    | "unsupported" ->
      let code_s = as_string (member "code" v) in
      DEUnsupported {
        node = as_string (member "node" v);
        code = (match unsupported_reason_of_name code_s with
                | Some c -> c
                | None   -> fail "unknown unsupported reason code '%s'" code_s);
      }
    | k -> fail "unknown deriv_entry '%s'" k
  )
  | _ -> fail "deriv_entry must be a single-key object"

let grad_map_to_json (m : (string * deriv_entry) list) : Yojson.Safe.t =
  obj (List.map (fun (p, de) -> (p, deriv_entry_to_json de)) m)

let grad_map_of_json = function
  | `Assoc pairs -> List.map (fun (name, de_j) -> (name, deriv_entry_of_json de_j)) pairs
  | `Null -> []
  | _ -> []

(* Append a grad_map field only when non-empty (mirrors the Rust
   `skip_serializing_if`, so an un-computed gradient serialises byte-identically). *)
let grad_field key (m : (string * deriv_entry) list) =
  match m with [] -> [] | _ -> [(key, grad_map_to_json m)]

(* A [diffable] serialises as the nested Rust `Diffable` shape:
     {"expr": <expr>}                              (grad empty)
     {"expr": <expr>, "grad": {param → deriv_entry}}   (grad non-empty)
   `expr` first, `grad` omitted when empty (Rust `skip_serializing_if`). A
   likelihood field `mean : diffable` therefore serialises as
   {"mean": {"expr": …, "grad": …}}. *)
let diffable_to_json (d : diffable) : Yojson.Safe.t =
  obj (("expr", expr_to_json d.expr) :: grad_field "grad" d.grad)

let diffable_of_json j : diffable =
  { expr = expr_of_json (member "expr" j);
    grad = (match member_opt "grad" j with
            | None | Some `Null -> []
            | Some g -> grad_map_of_json g) }

let draw_method_to_json (dm : draw_method) : Yojson.Safe.t =
  match dm with
  | DrawPoisson       -> str "poisson"
  | DrawDeterministic -> str "deterministic"
  (* Keep the σ² VALUE as the bare expression (byte-stable golden); carry the
     gradient as an adjacent sibling key present only when non-empty. *)
  | DrawOverdispersed { sigma_sq; sigma_sq_grad } ->
    obj ([("overdispersed", expr_to_json sigma_sq)]
         @ grad_field "overdispersed_grad" sigma_sq_grad)

let draw_method_of_json j =
  match j with
  | `String "poisson"       -> Ir.DrawPoisson
  | `String "deterministic" -> Ir.DrawDeterministic
  | `Assoc fields when List.mem_assoc "overdispersed" fields ->
    let sigma_sq = expr_of_json (List.assoc "overdispersed" fields) in
    let sigma_sq_grad = match List.assoc_opt "overdispersed_grad" fields with
      | None | Some `Null -> []
      | Some g -> grad_map_of_json g
    in
    Ir.DrawOverdispersed { sigma_sq; sigma_sq_grad }
  | _ -> fail "draw_method must be \"poisson\", \"deterministic\", or {\"overdispersed\": expr}"

let transition_lineage_to_json (l : transition_lineage) : Yojson.Safe.t =
  obj [
    ("is_lineage_event", bool l.is_lineage_event);
    ("parent_pool_weights",
     arr (List.map (fun (comp, e) -> arr [str comp; expr_to_json e])
            l.parent_pool_weights));
  ]

let transition_lineage_of_json j : transition_lineage =
  { is_lineage_event = as_bool (member "is_lineage_event" j);
    parent_pool_weights =
      List.map (fun pair ->
        match as_list pair with
        | [comp; e] -> (as_string comp, expr_of_json e)
        | _ -> fail "parent_pool_weights entry must be a 2-element [comp, expr] array")
        (as_list (member "parent_pool_weights" j));
  }

let transition_to_json (t : transition) : Yojson.Safe.t =
  obj (
    [ ("name",         str t.name);
      ("stoichiometry", arr (List.map stoich_entry_to_json t.stoichiometry));
      ("rate",         expr_to_json t.rate);
      ("metadata",     match t.metadata  with None -> null | Some m -> metadata_to_json m);
    ]
    @ (match t.draw_method with
       | DrawPoisson -> []
       | dm          -> [("draw_method", draw_method_to_json dm)])
    @ (match t.rate_grad with
       | [] -> []
       | grads -> [("rate_grad", obj (List.map (fun (p, e) -> (p, expr_to_json e)) grads))])
    @ (match t.lineage with
       | None   -> []
       | Some l -> [("lineage", transition_lineage_to_json l)])
  )

let transition_of_json j =
  { name         = as_string (member "name" j);
    stoichiometry = List.map stoich_entry_of_json (as_list (member "stoichiometry" j));
    rate         = expr_of_json (member "rate" j);
    metadata     = (match member_opt "metadata" j with
                    | None | Some `Null -> None
                    | Some m -> Some (metadata_of_json m));
    draw_method  = (match member_opt "draw_method" j with
                    | None | Some `Null -> Ir.DrawPoisson
                    | Some dm -> draw_method_of_json dm);
    rate_grad    = (match member_opt "rate_grad" j with
                    | None | Some `Null -> []
                    | Some (`Assoc pairs) ->
                      List.map (fun (name, expr_j) -> (name, expr_of_json expr_j)) pairs
                    | Some _ -> []);
    lineage      = (match member_opt "lineage" j with
                    | None | Some `Null -> None
                    | Some l -> Some (transition_lineage_of_json l));
  }

(* ── ODE equation ────────────────────────────────────────────────────────── *)

let ode_equation_to_json (e : ode_equation) : Yojson.Safe.t =
  obj [("compartment", str e.compartment); ("derivative", expr_to_json e.derivative)]

let ode_equation_of_json j =
  { compartment = as_string (member "compartment" j);
    derivative  = expr_of_json (member "derivative" j) }

(* ── Time functions ──────────────────────────────────────────────────────── *)

let time_func_kind_to_json (k : time_func_kind) : Yojson.Safe.t =
  match k with
  | Sinusoidal s ->
    obj [("sinusoidal", obj [
      ("amplitude", expr_to_json s.amplitude); ("period", expr_to_json s.period);
      ("phase",     expr_to_json s.phase);     ("baseline", expr_to_json s.baseline);
    ])]
  | Piecewise p ->
    obj [("piecewise", obj [
      ("breakpoints", arr (List.map expr_to_json p.breakpoints));
      ("values",      arr (List.map expr_to_json p.values));
    ])]
  | Interpolated i ->
    obj [("interpolated", obj [
      ("times",  arr (List.map expr_to_json i.times));
      ("values", arr (List.map expr_to_json i.values));
      ("method", str i.method_);
    ])]
  | Periodic p ->
    obj [("periodic", obj [
      ("period", expr_to_json p.period);
      ("values", arr (List.map expr_to_json p.values));
    ])]
  | Fourier f ->
    obj [("fourier", obj [
      ("period", expr_to_json f.period);
      ("harmonics", arr (List.map (fun (a, b) ->
        arr [expr_to_json a; expr_to_json b]) f.harmonics));
    ])]
  | PeriodicSpline ps ->
    obj [("periodic_spline", obj [
      ("period",  expr_to_json ps.period);
      ("n_basis", int ps.n_basis);
      ("degree",  int ps.degree);
      ("coefs",   arr (List.map expr_to_json ps.coefs));
    ])]

let time_func_kind_of_json j =
  match j with
  | `Assoc [(key, v)] -> (
    match key with
    | "sinusoidal" ->
      Sinusoidal {
        amplitude = expr_of_json (member "amplitude" v);
        period    = expr_of_json (member "period"    v);
        phase     = expr_of_json (member "phase"     v);
        baseline  = expr_of_json (member "baseline"  v);
      }
    | "piecewise" ->
      Piecewise {
        breakpoints = List.map expr_of_json (as_list (member "breakpoints" v));
        values      = List.map expr_of_json (as_list (member "values"      v));
      }
    | "interpolated" ->
      Interpolated {
        times   = List.map expr_of_json (as_list (member "times"  v));
        values  = List.map expr_of_json (as_list (member "values" v));
        method_ = as_string (member "method" v);
      }
    | "periodic" ->
      Periodic {
        period = expr_of_json (member "period" v);
        values = List.map expr_of_json (as_list (member "values" v));
      }
    | "fourier" ->
      let harm_pair = function
        | `List [a; b] -> (expr_of_json a, expr_of_json b)
        | _ -> fail "fourier harmonic must be a 2-element array [a, b]"
      in
      Fourier {
        period    = expr_of_json (member "period" v);
        harmonics = List.map harm_pair (as_list (member "harmonics" v));
      }
    | "periodic_spline" ->
      PeriodicSpline {
        period  = expr_of_json (member "period" v);
        n_basis = as_int (member "n_basis" v);
        degree  = as_int (member "degree" v);
        coefs   = List.map expr_of_json (as_list (member "coefs" v));
      }
    | k -> fail "unknown time_func_kind '%s'" k
  )
  | _ -> fail "time_func_kind must be a single-key object"

let time_function_to_json (tf : time_function) : Yojson.Safe.t =
  let (p, t) = tf.dim in
  (* gh#314: [lag] is omitted when [None] so a forcing without a lag
     serializes byte-identically to the pre-gh#314 wire format. *)
  obj (
    [ ("name", str tf.name);
      ("kind", time_func_kind_to_json tf.kind);
      ("dim",  arr [int p; int t]); ]
    @ opt_field "lag" expr_to_json tf.lag)

let time_function_of_json j =
  { name = as_string (member "name" j);
    kind = time_func_kind_of_json (member "kind" j);
    dim  = (match member "dim" j with
            | `List [p; t] -> (as_int p, as_int t)
            | _ -> fail "time_function.dim must be a two-element [P, T] array");
    lag  = (match member_opt "lag" j with
            | Some `Null | None -> None
            | Some v            -> Some (expr_of_json v)); }

(* ── Table ───────────────────────────────────────────────────────────────── *)

let oob_policy_to_json = function
  | Error -> str "error"

let oob_policy_of_json j =
  match as_string j with
  | "error" -> Error
  | s -> fail "unknown oob_policy '%s'" s

let table_to_json (t : table) : Yojson.Safe.t =
  let source_field = match t.source with
    | Inline vs  -> ("values",   arr (List.map expr_to_json vs))
    | External n -> ("external", str n)
  in
  let base = [
    ("name",          str t.name);
    source_field;
    ("out_of_bounds", oob_policy_to_json t.out_of_bounds);
  ] in
  let with_cell_kind = match t.cell_kind with
    | None   -> base
    | Some k -> base @ [("cell_kind", str k)]
  in
  obj with_cell_kind

let table_source_of_json j =
  match j with
  | `Assoc kvs when List.mem_assoc "external" kvs ->
    let name = as_string (List.assoc "external" kvs) in
    (Ir.External name : Ir.table_source)
  | _ ->
    (Ir.Inline (List.map expr_of_json (as_list (member "values" j))) : Ir.table_source)

let table_of_json j =
  { Ir.name          = as_string (member "name" j);
    Ir.source        = table_source_of_json j;
    Ir.out_of_bounds = oob_policy_of_json (member "out_of_bounds" j);
    Ir.cell_kind     = (match member_opt "cell_kind" j with
                        | Some `Null | None -> None
                        | Some k -> Some (as_string k)); }

(* ── Interventions ───────────────────────────────────────────────────────── *)

let intervention_schedule_to_json (s : intervention_schedule) : Yojson.Safe.t =
  match s with
  | AtTimes ts ->
    obj [("at_times", arr (List.map flt ts))]
  | AtTimesExpr exprs ->
    (* gh#69: parametric `at [...]`. JSON key is distinct from `at_times`
       so old IRs without expressions deserialize as AtTimes unchanged. *)
    obj [("at_times_expr", arr (List.map expr_to_json exprs))]
  | Recurring r ->
    obj [("recurring", obj (
      [("start",  flt r.start);
       ("period", flt r.period);
       ("end",    flt r.end_)]
      @ (match r.at_day with None -> [] | Some d -> [("at_day", flt d)])
    ))]

let intervention_schedule_of_json j =
  match j with
  | `Assoc [(key, v)] -> (
    match key with
    | "at_times"      -> AtTimes     (List.map as_float (as_list v))
    | "at_times_expr" -> AtTimesExpr (List.map expr_of_json (as_list v))
    | "recurring" ->
      Recurring {
        start  = as_float (member "start"  v);
        period = as_float (member "period" v);
        end_   = as_float (member "end"    v);
        at_day = (match member_opt "at_day" v with
                  | Some n -> Some (as_float n) | None -> None);
      }
    | k -> fail "unknown intervention_schedule '%s'" k
  )
  | _ -> fail "intervention_schedule must be a single-key object"

let action_to_json (a : action) : Yojson.Safe.t =
  match a with
  | FractionTransfer ft ->
    obj [("fraction_transfer", obj [
      ("src",      str ft.src);
      ("dst",      str ft.dst);
      ("fraction", expr_to_json ft.fraction);
    ])]
  | AbsoluteTransfer at_ ->
    obj [("absolute_transfer", obj [
      ("src",   str at_.src);
      ("dst",   str at_.dst);
      ("count", expr_to_json at_.count);
    ])]
  | Set sa ->
    obj [("set", obj [
      ("compartment", str sa.compartment);
      ("value",       expr_to_json sa.value);
    ])]
  | AddAction aa ->
    obj [("add", obj [
      ("compartment", str aa.add_compartment);
      ("count",       expr_to_json aa.add_count);
    ])]

let action_of_json j =
  match j with
  | `Assoc [(key, v)] -> (
    match key with
    | "fraction_transfer" ->
      FractionTransfer {
        src      = as_string (member "src" v);
        dst      = as_string (member "dst" v);
        fraction = expr_of_json (member "fraction" v);
      }
    | "absolute_transfer" ->
      AbsoluteTransfer {
        src   = as_string (member "src"   v);
        dst   = as_string (member "dst"   v);
        count = expr_of_json (member "count" v);
      }
    | "set" ->
      Set {
        compartment = as_string (member "compartment" v);
        value       = expr_of_json (member "value" v);
      }
    | "add" ->
      AddAction {
        add_compartment = as_string (member "compartment" v);
        add_count       = expr_of_json (member "count" v);
      }
    | k -> fail "unknown action '%s'" k
  )
  | _ -> fail "action must be a single-key object"

(* gh#204. Reactive fire source. Wire shapes mirror the Rust serde derives:
   AgendaScope is a snake_case unit enum (a bare string); ReactiveTrigger is a
   struct (cooldown skipped when absent); FireSource is externally tagged
   ({"scheduled": ..} / {"reactive": ..}). *)

(* gh#204. Reactive trigger predicate. Wire shapes mirror the Rust serde:
   CmpOp/ObsReducer are snake_case unit enums; TriggerQuantity / TriggerThreshold
   / TriggerExpr are externally tagged; And/Or carry a 2-element array. *)
let cmp_op_to_json = function
  | CmpLt -> str "lt" | CmpLe -> str "le" | CmpGt -> str "gt"
  | CmpGe -> str "ge" | CmpEq -> str "eq" | CmpNeq -> str "neq"

let cmp_op_of_json j =
  match as_string j with
  | "lt" -> CmpLt | "le" -> CmpLe | "gt" -> CmpGt
  | "ge" -> CmpGe | "eq" -> CmpEq | "neq" -> CmpNeq
  | s -> fail "unknown cmp_op '%s'" s

let obs_reducer_to_json = function
  | RedLatest -> str "latest" | RedSum -> str "sum"
  | RedMean -> str "mean" | RedMax -> str "max"

let obs_reducer_of_json j =
  match as_string j with
  | "latest" -> RedLatest | "sum" -> RedSum
  | "mean" -> RedMean | "max" -> RedMax
  | s -> fail "unknown obs_reducer '%s'" s

let trigger_quantity_to_json (q : trigger_quantity) : Yojson.Safe.t =
  match q with
  | TQObserved { stream; window; reducer } ->
    obj [("observed", obj (
      [("stream", str stream)]
      @ (match window with None -> [] | Some w -> [("window", flt w)])
      @ [("reducer", obs_reducer_to_json reducer)]
    ))]

let trigger_quantity_of_json j =
  match j with
  | `Assoc [("observed", v)] ->
    TQObserved {
      stream  = as_string (member "stream" v);
      window  = (match member_opt "window" v with
                 | Some n -> Some (as_float n) | None -> None);
      reducer = obs_reducer_of_json (member "reducer" v);
    }
  | _ -> fail "trigger_quantity must be a single-key {\"observed\": ..} object"

let trigger_threshold_to_json (t : trigger_threshold) : Yojson.Safe.t =
  match t with
  | TTConst v -> obj [("const", flt v)]
  | TTParam s -> obj [("param", str s)]

let trigger_threshold_of_json j =
  match j with
  | `Assoc [("const", v)] -> TTConst (as_float v)
  | `Assoc [("param", v)] -> TTParam (as_string v)
  | _ -> fail "trigger_threshold must be {\"const\": ..} or {\"param\": ..}"

let rec trigger_expr_to_json (e : trigger_expr) : Yojson.Safe.t =
  match e with
  | TECmp (lhs, op, rhs) ->
    obj [("cmp", obj [
      ("lhs", trigger_quantity_to_json lhs);
      ("op",  cmp_op_to_json op);
      ("rhs", trigger_threshold_to_json rhs);
    ])]
  | TEAnd (a, b) -> obj [("and", arr [trigger_expr_to_json a; trigger_expr_to_json b])]
  | TEOr  (a, b) -> obj [("or",  arr [trigger_expr_to_json a; trigger_expr_to_json b])]
  | TENot a      -> obj [("not", trigger_expr_to_json a)]

let rec trigger_expr_of_json j =
  match j with
  | `Assoc [("cmp", v)] ->
    TECmp (trigger_quantity_of_json (member "lhs" v),
           cmp_op_of_json            (member "op"  v),
           trigger_threshold_of_json (member "rhs" v))
  | `Assoc [("and", `List [a; b])] -> TEAnd (trigger_expr_of_json a, trigger_expr_of_json b)
  | `Assoc [("or",  `List [a; b])] -> TEOr  (trigger_expr_of_json a, trigger_expr_of_json b)
  | `Assoc [("not", a)]            -> TENot (trigger_expr_of_json a)
  | _ -> fail "unknown trigger_expr (expected one of cmp/and/or/not)"

let reactive_trigger_to_json (t : reactive_trigger) : Yojson.Safe.t =
  obj (
    [ ("when",  trigger_expr_to_json t.when_);
      ("after", flt t.after);
      ("once",  bool t.once) ]
    @ (match t.cooldown with None -> [] | Some c -> [("cooldown", flt c)])
  )

let reactive_trigger_of_json j =
  { when_    = trigger_expr_of_json (member "when" j);
    after    = as_float (member "after" j);
    once     = as_bool (member "once" j);
    cooldown = (match member_opt "cooldown" j with
                | Some n -> Some (as_float n) | None -> None); }

let fire_source_to_json (f : fire_source) : Yojson.Safe.t =
  match f with
  | Scheduled s -> obj [("scheduled", intervention_schedule_to_json s)]
  | Reactive  t -> obj [("reactive",  reactive_trigger_to_json t)]

let fire_source_of_json j =
  match j with
  | `Assoc [(key, v)] -> (
    match key with
    | "scheduled" -> Scheduled (intervention_schedule_of_json v)
    | "reactive"  -> Reactive  (reactive_trigger_of_json v)
    | k -> fail "unknown fire_source '%s'" k
  )
  | _ -> fail "fire_source must be a single-key object"

let intervention_to_json (iv : intervention) : Yojson.Safe.t =
  obj (
    [("name", str iv.name)]
    @ (match iv.base_name with None -> [] | Some s -> [("base_name", str s)])
    @ [ ("fire",    fire_source_to_json iv.fire);
        ("actions", arr (List.map action_to_json iv.actions)); ]
    (* Skip-emit the default (Scenario), mirroring the former
       always_active skip-false discipline: scenario interventions carry no
       key, events carry `"kind": "event"`. *)
    @ (match iv.kind with Event -> [("kind", str "event")] | Scenario -> [])
  )

let intervention_of_json j =
  { name      = as_string (member "name" j);
    base_name = (match member_opt "base_name" j with
                 | Some (`String s) -> Some s
                 | _ -> None);
    fire      = fire_source_of_json (member "fire" j);
    actions   = List.map action_of_json (as_list (member "actions" j));
    kind      = (match member_opt "kind" j with
                 | Some (`String "event")    -> Event
                 | Some (`String "scenario") -> Scenario
                 | Some `Null | None         -> Scenario
                 | Some _ ->
                   fail "intervention '%s': kind must be \"scenario\" or \"event\""
                     (as_string (member "name" j)));
  }

(* ── Observation model ───────────────────────────────────────────────────── *)

let projection_to_json (p : projection) : Yojson.Safe.t =
  match p with
  | CumulativeFlow    tn -> obj [("cumulative_flow",     str tn)]
  | CumulativeFlowSum fs -> obj [("cumulative_flow_sum", arr (List.map str fs))]
  | CurrentPop        cn -> obj [("current_pop",         str cn)]
  | CurrentPopSum     cs -> obj [("current_pop_sum",     arr (List.map str cs))]
  | DerivedExpr       e  -> obj [("derived_expr",        expr_to_json e)]

let projection_of_json j =
  match j with
  | `Assoc [(key, v)] -> (
    match key with
    | "cumulative_flow"     -> CumulativeFlow    (as_string v)
    | "cumulative_flow_sum" -> CumulativeFlowSum (List.map as_string (as_list v))
    | "current_pop"         -> CurrentPop        (as_string v)
    | "current_pop_sum"     -> CurrentPopSum     (List.map as_string (as_list v))
    | "derived_expr"        -> DerivedExpr       (expr_of_json v)
    | k -> fail "unknown projection '%s'" k
  )
  | _ -> fail "projection must be a single-key object"

(* Each differentiable argument is a [diffable] and serialises as the nested
   `{"expr": …, "grad": …}` shape (grad omitted when empty), matching the Rust
   field name / declaration order. `n` (Binomial/BetaBinomial) is a bare expr —
   θ-independent, no grad. *)
let likelihood_to_json (l : likelihood) : Yojson.Safe.t =
  match l with
  | Poisson p ->
    obj [("poisson", obj [("rate", diffable_to_json p.rate)])]
  | NegBinomial nb ->
    obj [("neg_binomial", obj [
      ("mean", diffable_to_json nb.mean);
      ("dispersion", diffable_to_json nb.dispersion)])]
  | Normal n ->
    obj [("normal", obj [
      ("mean", diffable_to_json n.mean);
      ("sd", diffable_to_json n.sd)])]
  | Binomial b ->
    obj [("binomial", obj [
      ("n", expr_to_json b.n); ("p", diffable_to_json b.p)])]
  | BetaBinomial bb ->
    obj [("beta_binomial", obj [
      ("n", expr_to_json bb.n);
      ("alpha", diffable_to_json bb.alpha);
      ("beta",  diffable_to_json bb.beta)])]
  | Bernoulli b ->
    obj [("bernoulli", obj [("p", diffable_to_json b.p)])]

let likelihood_of_json j =
  let d key v = diffable_of_json (member key v) in
  match j with
  | `Assoc [(key, v)] -> (
    match key with
    | "poisson" ->
      Poisson { rate = d "rate" v }
    | "neg_binomial" ->
      NegBinomial { mean = d "mean" v; dispersion = d "dispersion" v }
    | "normal" ->
      Normal { mean = d "mean" v; sd = d "sd" v }
    | "binomial" ->
      Binomial { n = expr_of_json (member "n" v); p = d "p" v }
    | "beta_binomial" ->
      BetaBinomial {
        n = expr_of_json (member "n" v);
        alpha = d "alpha" v;
        beta  = d "beta"  v;
      }
    | "bernoulli" ->
      Bernoulli { p = d "p" v }
    | k -> fail "unknown likelihood '%s'" k
  )
  | _ -> fail "likelihood must be a single-key object"

let obs_schedule_to_json (s : observation_schedule) : Yojson.Safe.t =
  match s with
  | ObsAtTimes ts ->
    obj [("at_times", arr (List.map flt ts))]
  | ObsRegular r ->
    obj [("regular", obj [
      ("start", flt r.start);
      ("step",  flt r.step);
      ("end",   flt r.end_);
    ])]

let obs_schedule_of_json j =
  match j with
  | `Assoc [(key, v)] -> (
    match key with
    | "at_times" -> ObsAtTimes (List.map as_float (as_list v))
    | "regular"  ->
      ObsRegular {
        start = as_float (member "start" v);
        step  = as_float (member "step"  v);
        end_  = as_float (member "end"   v);
      }
    | k -> fail "unknown observation_schedule '%s'" k
  )
  | _ -> fail "observation_schedule must be a single-key object"

let obs_column_role_to_json (r : obs_column_role) : Yojson.Safe.t =
  match r with
  | RoleTime    -> `String "time"
  | RoleDim d   -> obj [("dim", str d)]
  | RoleValue k -> obj [("value", str (param_kind_name k))]

let obs_column_role_of_json j =
  match j with
  | `String "time" -> RoleTime
  | `Assoc _ -> (
    match member_opt "dim" j, member_opt "value" j with
    | Some d, None -> RoleDim (as_string d)
    | None, Some v ->
      (match param_kind_of_name (as_string v) with
       | Some k -> RoleValue k
       | None   -> fail "unknown column value type '%s'" (as_string v))
    | _ -> fail "column role object must be {dim:…} or {value:…}")
  | _ -> fail "column role must be \"time\" or {dim:…}/{value:…}"

let obs_column_to_json (c : obs_column) : Yojson.Safe.t =
  obj [
    ("name", str c.col_name);
    ("role", obs_column_role_to_json c.col_role);
  ]

let obs_column_of_json j =
  { col_name = as_string (member "name" j);
    col_role = obs_column_role_of_json (member "role" j); }

let stratum_key_to_json ((d, l) : string * string) : Yojson.Safe.t =
  obj [("dim", str d); ("level", str l)]

let stratum_key_of_json j =
  (as_string (member "dim" j), as_string (member "level" j))

let observation_model_to_json (om : observation_model) : Yojson.Safe.t =
  let base = [
    ("name",          str om.name);
    ("source",        str om.obs_source);
    ("columns",       arr (List.map obs_column_to_json om.columns));
    ("scored",        str om.scored);
  ] in
  let sched = match om.emit_schedule with
    | None   -> []
    | Some s -> [("emit_schedule", obs_schedule_to_json s)]
  in
  (* Omit the `stratum` key when empty (unstratified stream) — mirrors the
     Rust `skip_serializing_if = "Vec::is_empty"`, so existing goldens for
     un-indexed observations are byte-identical. *)
  let stratum = match om.stratum with
    | [] -> []
    | ss -> [("stratum", arr (List.map stratum_key_to_json ss))]
  in
  obj (base @ sched @ stratum @ [
    ("projection",  projection_to_json om.projection);
    ("likelihood",  likelihood_to_json om.likelihood);
  ])

let observation_model_of_json j =
  { name          = as_string (member "name"   j);
    obs_source    = as_string (member "source" j);
    columns       = List.map obs_column_of_json (as_list (member "columns" j));
    scored        = as_string (member "scored" j);
    emit_schedule = (match member_opt "emit_schedule" j with
                     | Some `Null | None -> None
                     | Some s -> Some (obs_schedule_of_json s));
    stratum       = (match member_opt "stratum" j with
                     | Some `Null | None -> []
                     | Some s -> List.map stratum_key_of_json (as_list s));
    projection    = projection_of_json  (member "projection" j);
    likelihood    = likelihood_of_json  (member "likelihood" j);
  }

(* ── Generated quantities ─────────────────────────────────────────────────────
   Mirror of rust/crates/ir/src/quantity.rs. Externally-tagged single-key
   objects throughout; `state` reuses the shared expr encoding, reduction
   arithmetic reuses the shared bin_op/un_op encoding, and stratum reuses
   stratum_key_to_json (omitted when empty, like observation_model). *)

let qref_to_json (q : qref) : Yojson.Safe.t =
  obj (
    [("name", str q.qref_name)]
    @ (match q.qref_stratum with
       | [] -> []
       | ss -> [("stratum", arr (List.map stratum_key_to_json ss))])
  )

let qref_of_json j : qref =
  { qref_name    = as_string (member "name" j);
    qref_stratum = (match member_opt "stratum" j with
                    | Some `Null | None -> []
                    | Some s -> List.map stratum_key_of_json (as_list s)); }

let rec scalar_expr_to_json (se : scalar_expr) : Yojson.Safe.t =
  match se with
  | SConst v -> obj [("const", flt v)]
  | SParam p -> obj [("param", str p)]
  | SQRef  q -> obj [("q_ref", qref_to_json q)]
  | SUnOp { op; arg } ->
    obj [("un_op", obj [
      ("op",  str (un_op_str op));
      ("arg", scalar_expr_to_json arg);
    ])]
  | SBinOp { op; left; right } ->
    obj [("bin_op", obj [
      ("op",    str (bin_op_str op));
      ("left",  scalar_expr_to_json left);
      ("right", scalar_expr_to_json right);
    ])]
  | SCond { pred; then_; else_ } ->
    obj [("cond", obj [
      ("pred", scalar_expr_to_json pred);
      ("then", scalar_expr_to_json then_);
      ("else", scalar_expr_to_json else_);
    ])]

let rec scalar_expr_of_json j : scalar_expr =
  match j with
  | `Assoc [("const", v)] -> SConst (as_float v)
  | `Assoc [("param", v)] -> SParam (as_string v)
  | `Assoc [("q_ref", v)] -> SQRef (qref_of_json v)
  | `Assoc [("un_op", v)] ->
    SUnOp { op  = un_op_of_str (as_string (member "op" v));
            arg = scalar_expr_of_json (member "arg" v); }
  | `Assoc [("bin_op", v)] ->
    SBinOp { op    = bin_op_of_str (as_string (member "op" v));
             left  = scalar_expr_of_json (member "left" v);
             right = scalar_expr_of_json (member "right" v); }
  | `Assoc [("cond", v)] ->
    SCond { pred  = scalar_expr_of_json (member "pred" v);
            then_ = scalar_expr_of_json (member "then" v);
            else_ = scalar_expr_of_json (member "else" v); }
  | _ -> fail "scalar_expr must be a single-key object \
               (const/param/q_ref/un_op/bin_op/cond)"

let value_reduce_to_json (v : value_reduce) : Yojson.Safe.t =
  match v with
  | VFinal -> str "final"
  | VMax   -> str "max"
  | VMin   -> str "min"
  | VMean  -> str "mean"
  | VCountAbove e -> obj [("count_above", expr_to_json e)]
  | VCountBelow e -> obj [("count_below", expr_to_json e)]

let value_reduce_of_json j : value_reduce =
  match j with
  | `String "final" -> VFinal
  | `String "max"   -> VMax
  | `String "min"   -> VMin
  | `String "mean"  -> VMean
  | `Assoc [("count_above", e)] -> VCountAbove (expr_of_json e)
  | `Assoc [("count_below", e)] -> VCountBelow (expr_of_json e)
  | _ -> fail "unknown value_reduce \
               (expected final/max/min/mean or count_above/count_below)"

let time_reduce_to_json (t : time_reduce) : Yojson.Safe.t =
  match t with
  | TimeOfMax    -> str "time_of_max"
  | TimeOfMin    -> str "time_of_min"
  | FirstAbove e -> obj [("first_above", expr_to_json e)]
  | FirstBelow e -> obj [("first_below", expr_to_json e)]
  | LastAbove  e -> obj [("last_above",  expr_to_json e)]
  | LastBelow  e -> obj [("last_below",  expr_to_json e)]

let time_reduce_of_json j : time_reduce =
  match j with
  | `String "time_of_max" -> TimeOfMax
  | `String "time_of_min" -> TimeOfMin
  | `Assoc [("first_above", e)] -> FirstAbove (expr_of_json e)
  | `Assoc [("first_below", e)] -> FirstBelow (expr_of_json e)
  | `Assoc [("last_above",  e)] -> LastAbove  (expr_of_json e)
  | `Assoc [("last_below",  e)] -> LastBelow  (expr_of_json e)
  | _ -> fail "unknown time_reduce \
               (expected time_of_max/time_of_min or first_/last_above/below)"

let temporal_reduce_to_json (r : temporal_reduce) : Yojson.Safe.t =
  match r with
  | RValue v  -> obj [("value", value_reduce_to_json v)]
  | RTime  t  -> obj [("time",  time_reduce_to_json t)]
  | RIntegral -> str "integral"

let temporal_reduce_of_json j : temporal_reduce =
  match j with
  | `String "integral"    -> RIntegral
  | `Assoc [("value", v)] -> RValue (value_reduce_of_json v)
  | `Assoc [("time",  t)] -> RTime  (time_reduce_of_json t)
  | _ -> fail "temporal_reduce must be \"integral\" or {\"value\":..}/{\"time\":..}"

let quantity_source_to_json (s : quantity_source) : Yojson.Safe.t =
  match s with
  | QSState e        -> obj [("state", expr_to_json e)]
  | QSObservation st -> obj [("observation", obj [("stream", str st)])]

let quantity_source_of_json j : quantity_source =
  match j with
  | `Assoc [("state", e)]       -> QSState (expr_of_json e)
  | `Assoc [("observation", v)] -> QSObservation (as_string (member "stream" v))
  | _ -> fail "quantity_source must be {\"state\": expr} or {\"observation\": {\"stream\": ..}}"

let quantity_body_to_json (b : quantity_body) : Yojson.Safe.t =
  match b with
  | QBReduced { source; reduce } ->
    obj [("reduced", obj (
      [("source", quantity_source_to_json source)]
      @ (match reduce with
         | None   -> []
         | Some r -> [("reduce", temporal_reduce_to_json r)])
    ))]
  | QBDerived se -> obj [("derived", scalar_expr_to_json se)]

let quantity_body_of_json j : quantity_body =
  match j with
  | `Assoc [("reduced", v)] ->
    QBReduced {
      source = quantity_source_of_json (member "source" v);
      reduce = (match member_opt "reduce" v with
                | Some `Null | None -> None
                | Some r -> Some (temporal_reduce_of_json r));
    }
  | `Assoc [("derived", se)] -> QBDerived (scalar_expr_of_json se)
  | _ -> fail "quantity_body must be {\"reduced\":..} or {\"derived\":..}"

let quantity_to_json (q : quantity) : Yojson.Safe.t =
  obj (
    [("name", str q.q_name)]
    @ (match q.q_stratum with
       | [] -> []
       | ss -> [("stratum", arr (List.map stratum_key_to_json ss))])
    @ [("body", quantity_body_to_json q.q_body)]
    @ (match q.q_dimension with
       | None -> []
       | Some (p, t) -> [("dimension", arr [int p; int t])])
  )

let quantity_of_json j : quantity =
  { q_name    = as_string (member "name" j);
    q_stratum = (match member_opt "stratum" j with
                 | Some `Null | None -> []
                 | Some s -> List.map stratum_key_of_json (as_list s));
    q_body    = quantity_body_of_json (member "body" j);
    q_dimension = (match member_opt "dimension" j with
                   | Some `Null | None -> None
                   | Some d -> (match as_list d with
                                | [p; t] -> Some (as_int p, as_int t)
                                | _ -> fail "quantity dimension must be [p, t]")); }

(* ── Counterfactual contrasts ──────────────────────────────────────────────────
   Mirror of rust/crates/ir/src/contrast.rs. Externally-tagged single-key objects;
   the run-member sub-namespace serializes as "quantities"/"observations" and the
   binop reuses the shared bin_op encoding. *)

let run_namespace_to_json (ns : run_namespace) : Yojson.Safe.t =
  match ns with
  | NsQuantities   -> str "quantities"
  | NsObservations -> str "observations"

let run_namespace_of_json j : run_namespace =
  match as_string j with
  | "quantities"   -> NsQuantities
  | "observations" -> NsObservations
  | s -> fail "run namespace must be \"quantities\" or \"observations\", got %S" s

let rec contrast_expr_to_json (ce : contrast_expr) : Yojson.Safe.t =
  match ce with
  | CRunMember { run; ns; member } ->
    obj [("run_member", obj [
      ("run",    str run);
      ("ns",     run_namespace_to_json ns);
      ("member", str member);
    ])]
  | CBinOp { op; left; right } ->
    obj [("bin_op", obj [
      ("op",    str (bin_op_str op));
      ("left",  contrast_expr_to_json left);
      ("right", contrast_expr_to_json right);
    ])]

let rec contrast_expr_of_json j : contrast_expr =
  match j with
  | `Assoc [("run_member", v)] ->
    CRunMember { run    = as_string (member "run" v);
                 ns     = run_namespace_of_json (member "ns" v);
                 member = as_string (member "member" v) }
  | `Assoc [("bin_op", v)] ->
    CBinOp { op    = bin_op_of_str (as_string (member "op" v));
             left  = contrast_expr_of_json (member "left" v);
             right = contrast_expr_of_json (member "right" v) }
  | _ -> fail "contrast_expr must be a single-key object (run_member/bin_op)"

let contrast_to_json (c : contrast) : Yojson.Safe.t =
  obj [
    ("name",   str c.c_name);
    ("body",   contrast_expr_to_json c.c_body);
  ]

let contrast_of_json j : contrast =
  { c_name   = as_string (member "name" j);
    c_body   = contrast_expr_of_json (member "body" j); }

(* ── Parameters ──────────────────────────────────────────────────────────── *)

let prior_dist_to_json (p : prior_dist) : Yojson.Safe.t =
  match p with
  | Uniform u ->
    obj [("uniform", obj [("lower", flt u.lower); ("upper", flt u.upper)])]
  | Normal_p n ->
    obj [("normal",  obj [("mean", flt n.mean); ("sd", flt n.sd)])]
  | LogNormal ln ->
    obj [("log_normal", obj [("mu", flt ln.mu); ("sigma", flt ln.sigma)])]
  | HalfNormal hn ->
    obj [("half_normal", obj [("sigma", flt hn.sigma)])]
  | Beta b ->
    obj [("beta", obj [("alpha", flt b.alpha); ("beta", flt b.beta)])]
  | Gamma g ->
    obj [("gamma", obj [("shape", flt g.shape); ("rate", flt g.rate)])]
  | Exponential e ->
    obj [("exponential", obj [("rate", flt e.rate)])]
  | LogUniform lu ->
    obj [("log_uniform", obj [("lower", flt lu.lu_lower); ("upper", flt lu.lu_upper)])]
  | TruncatedNormal tn ->
    obj [("truncated_normal", obj [
      ("mean",  flt tn.tn_mean);
      ("sd",    flt tn.tn_sd);
      ("lower", flt tn.tn_lower);
      ("upper", flt tn.tn_upper)])]
  | Fixed v ->
    obj [("fixed", flt v)]

let prior_dist_of_json j =
  match j with
  | `Assoc [(key, v)] -> (
    match key with
    | "uniform"     -> Uniform     { lower = as_float (member "lower" v); upper = as_float (member "upper" v) }
    | "normal"      -> Normal_p    { mean  = as_float (member "mean"  v); sd    = as_float (member "sd"    v) }
    | "log_normal"  -> LogNormal   { mu    = as_float (member "mu"    v); sigma = as_float (member "sigma" v) }
    | "half_normal" -> HalfNormal  { sigma = as_float (member "sigma" v) }
    | "beta"        -> Beta        { alpha = as_float (member "alpha" v); beta  = as_float (member "beta"  v) }
    | "gamma"       -> Gamma       { shape = as_float (member "shape" v); rate  = as_float (member "rate"  v) }
    | "exponential" -> Exponential { rate  = as_float (member "rate"  v) }
    | "log_uniform" -> LogUniform { lu_lower = as_float (member "lower" v); lu_upper = as_float (member "upper" v) }
    | "truncated_normal" -> TruncatedNormal {
        tn_mean  = as_float (member "mean"  v);
        tn_sd    = as_float (member "sd"    v);
        tn_lower = as_float (member "lower" v);
        tn_upper = as_float (member "upper" v) }
    | "fixed"       -> Fixed (as_float v)
    | k -> fail "unknown prior_dist '%s'" k
  )
  | _ -> fail "prior_dist must be a single-key object"

let transform_to_json = function
  | Log      -> str "log"
  | Logit    -> str "logit"
  | Identity -> str "identity"

let transform_of_json j =
  match as_string j with
  | "log" -> Log | "logit" -> Logit | "identity" -> Identity
  | s -> fail "unknown transform '%s'" s

let hierarchical_prior_to_json (h : hierarchical_prior) : Yojson.Safe.t =
  obj [
    ("kind",       str (hierarchical_kind_name h.hkind));
    ("args",       obj (List.map (fun (k, e) -> (k, expr_to_json e)) h.hargs));
    ("pool_over",  str h.hpool_over);
  ]

let hierarchical_prior_of_json j : hierarchical_prior =
  let kind_str = as_string (member "kind" j) in
  {
    hkind      = (match hierarchical_kind_of_name kind_str with
                  | exception Failure _ ->
                    fail "unknown hierarchical prior kind '%s'" kind_str
                  | k -> k);
    hargs      = (match member "args" j with
                  | `Assoc kvs -> List.map (fun (k, v) -> (k, expr_of_json v)) kvs
                  | _ -> fail "hierarchical prior args must be an object");
    hpool_over = as_string (member "pool_over" j);
  }

let prior_spec_to_json (p : prior_spec) : Yojson.Safe.t =
  match p with
  | Flat           -> str "flat"
  | Dist d         -> obj [("dist",         prior_dist_to_json d)]
  | Hierarchical h -> obj [("hierarchical", hierarchical_prior_to_json h)]

let prior_spec_of_json j : prior_spec =
  match j with
  | `String "flat"                -> Flat
  | `Assoc [("dist", v)]          -> Dist (prior_dist_of_json v)
  | `Assoc [("hierarchical", v)]  -> Hierarchical (hierarchical_prior_of_json v)
  | _ -> fail "prior_spec must be \"flat\" or a single-key {dist|hierarchical} object"

let param_value_to_json (v : param_value) : Yojson.Safe.t =
  match v with
  | Fixed f -> obj [("mode", str "fixed"); ("value", flt f)]
  | Required -> obj [("mode", str "required")]
  | Estimated e ->
    obj (
      [("mode", str "estimated")]
      @ (match e.est_init   with None -> [] | Some v -> [("init", flt v)])
      @ (match e.est_bounds with None -> [] | Some (lo, hi) -> [("bounds", arr [flt lo; flt hi])])
      @ [ ("prior",     prior_spec_to_json e.est_prior);
          ("transform", transform_to_json  e.est_transform); ]
    )

let param_value_of_json j : param_value =
  match member "mode" j with
  | `String "fixed"    -> Fixed (as_float (member "value" j))
  | `String "required" -> Required
  | `String "estimated" ->
    Estimated {
      est_init      = (match member_opt "init"   j with Some `Null | None -> None | Some v -> Some (as_float v));
      est_bounds    = (match member_opt "bounds" j with
        | Some `Null | None -> None
        | Some (`List [lo; hi]) -> Some (as_float lo, as_float hi)
        | _ -> fail "bounds must be a two-element array [lo, hi]");
      est_prior     = prior_spec_of_json (member "prior" j);
      est_transform = transform_of_json  (member "transform" j);
    }
  | _ -> fail "parameter value: \"mode\" must be \"fixed\", \"estimated\", or \"required\""

(* A `#'` doc block. Serialized omit-when-None so an undocumented parameter is
   byte-identical to before this field existed (golden-neutral). *)
let doc_to_json (d : doc) : Yojson.Safe.t =
  obj (
    (match d.text      with None -> [] | Some t -> [("text",   str t)]) @
    (match d.symbol    with None -> [] | Some s -> [("symbol", str s)]) @
    (match d.reference with None -> [] | Some r -> [("ref",    str r)]))

let doc_of_json j =
  let s key = match member_opt key j with Some (`String v) -> Some v | _ -> None in
  { text = s "text"; symbol = s "symbol"; reference = s "ref" }

(* The doc dictionary, serialized as `{ category: { name: doc, … }, … }`.
   Empty categories are omitted; an entirely-empty index serializes to `{}`,
   and the envelope omits the `docs` key altogether (see envelope_to_json). *)
let doc_index_to_json (di : doc_index) : Yojson.Safe.t =
  let category name entries =
    if entries = [] then []
    else [(name, obj (List.map (fun (k, d) -> (k, doc_to_json d)) entries))]
  in
  obj (
    category "parameters"   di.di_parameters @
    category "compartments" di.di_compartments @
    category "transitions"  di.di_transitions @
    category "observations" di.di_observations @
    category "dimensions"   di.di_dimensions @
    category "quantities"   di.di_quantities)

let doc_index_of_json j : doc_index =
  let category name = match member_opt name j with
    | Some (`Assoc kvs) -> List.map (fun (k, v) -> (k, doc_of_json v)) kvs
    | _ -> []
  in
  { di_parameters   = category "parameters";
    di_compartments = category "compartments";
    di_transitions  = category "transitions";
    di_observations = category "observations";
    di_dimensions   = category "dimensions";
    di_quantities   = category "quantities"; }

let doc_index_is_empty (di : doc_index) : bool =
  di.di_parameters = [] && di.di_compartments = [] && di.di_transitions = []
  && di.di_observations = [] && di.di_dimensions = [] && di.di_quantities = []

let parameter_to_json (p : parameter) : Yojson.Safe.t =
  obj [
    ("name",       str p.name);
    ("value",      param_value_to_json p.value);
    ("param_kind", match p.param_kind with None -> null | Some k -> str (param_kind_name k));
    ("param_dim",  match p.param_dim  with None -> null | Some (p_exp, t_exp) -> arr [int p_exp; int t_exp]);
  ]

let parameter_of_json j =
  { name       = as_string (member "name" j);
    value      = param_value_of_json (member "value" j);
    param_kind = (match member_opt "param_kind" j with
      | Some `Null | None -> None
      | Some k -> (match param_kind_of_name (as_string k) with
                   | Some pk -> Some pk
                   | None    -> fail "unknown param_kind '%s' (expected one of \
                                      rate|probability|count|positive|real|instant|duration)"
                                  (as_string k)));
    param_dim  = (match member_opt "param_dim" j with
      | Some (`List [p; t]) -> Some (as_int p, as_int t)
      | _ -> None);
  }

(* ── Initial conditions ──────────────────────────────────────────────────── *)

let initial_conditions_to_json (ic : initial_conditions) : Yojson.Safe.t =
  match ic with
  | Explicit kvs ->
    obj [("explicit", obj (List.map (fun (k, v) -> (k, flt v)) kvs))]
  | Parameterized kvs ->
    obj [("parameterized", obj (List.map (fun (k, e) -> (k, expr_to_json e)) kvs))]
  | FromDistribution kvs ->
    obj [("from_distribution", obj (List.map (fun (k, p) -> (k, prior_dist_to_json p)) kvs))]

let initial_conditions_of_json j =
  match j with
  | `Assoc [(key, v)] -> (
    match key with
    | "explicit" ->
      Explicit (List.map (fun (k, vv) -> (k, as_float vv)) (as_assoc v))
    | "parameterized" ->
      Parameterized (List.map (fun (k, vv) -> (k, expr_of_json vv)) (as_assoc v))
    | "from_distribution" ->
      FromDistribution (List.map (fun (k, vv) -> (k, prior_dist_of_json vv)) (as_assoc v))
    | k -> fail "unknown initial_conditions kind '%s'" k
  )
  | _ -> fail "initial_conditions must be a single-key object"

(* ── Output ──────────────────────────────────────────────────────────────── *)

let output_schedule_to_json (s : output_schedule) : Yojson.Safe.t =
  match s with
  | OutRegular r ->
    obj [("regular", obj [
      ("start", flt r.start);
      ("step",  flt r.step);
    ])]
  | OutAtTimes ts ->
    obj [("at_times", arr (List.map flt ts))]

let output_schedule_of_json j =
  match j with
  | `Assoc [(key, v)] -> (
    match key with
    | "regular" ->
      OutRegular {
        start = as_float (member "start" v);
        step  = as_float (member "step"  v);
      }
    | "at_times" -> OutAtTimes (List.map as_float (as_list v))
    | k -> fail "unknown output_schedule '%s'" k
  )
  | _ -> fail "output_schedule must be a single-key object"

let output_config_to_json (o : output_config) : Yojson.Safe.t =
  obj [
    ("times",        output_schedule_to_json o.times);
    ("format",       str o.format);
    ("trajectory",   bool o.trajectory);
    ("observations", bool o.observations);
  ]

let output_config_of_json j =
  { times        = output_schedule_of_json (member "times"        j);
    format       = as_string               (member "format"       j);
    trajectory   = as_bool                 (member "trajectory"   j);
    observations = as_bool                 (member "observations" j);
  }

(* ── Simulation config ───────────────────────────────────────────────────── *)

(* gh#166: integrator serialized internally-tagged — {"method":"rk4"} /
   {"method":"rk45","atol":…,"rtol":…} — mirroring the Rust enum. *)
let integrator_to_json (i : integrator) : Yojson.Safe.t =
  match i with
  | Rk4 -> obj [ ("method", str "rk4") ]
  | Rk45 { atol; rtol } ->
    obj (
      [ ("method", str "rk45") ]
      @ (match atol with None -> [] | Some v -> [ ("atol", flt v) ])
      @ (match rtol with None -> [] | Some v -> [ ("rtol", flt v) ])
    )

let integrator_of_json j =
  match member_opt "method" j with
  | Some (`String "rk45") ->
    Rk45 {
      atol = (match member_opt "atol" j with Some `Null | None -> None | Some v -> Some (as_float v));
      rtol = (match member_opt "rtol" j with Some `Null | None -> None | Some v -> Some (as_float v));
    }
  | Some (`String "rk4") -> Rk4
  (* Reject an unknown tag instead of silently defaulting to rk4 — mirrors the
     Rust internally-tagged enum, which hard-errors on an unknown method. *)
  | Some (`String s) -> fail "unknown integrator method '%s': expected \"rk4\" or \"rk45\"" s
  | Some _           -> fail "integrator: \"method\" must be a string"
  | None             -> fail "integrator: missing \"method\" field"

let simulation_config_to_json (s : simulation_config) : Yojson.Safe.t =
  (* integrator OMITTED at the Rk4 default, mirroring the Rust side's
     `skip_serializing_if`, so a default model's IR body is unchanged by gh#166
     (only the version string moves). *)
  obj (
    [ ("t_start",        flt s.t_start);
      ("t_end",          flt s.t_end);
      ("time_semantics", str s.time_semantics);
      ("dt",             match s.dt       with None -> null | Some v -> flt v);
      ("rng_seed",       match s.rng_seed with None -> null | Some n -> int n);
    ]
    @ (match s.integrator with Rk4 -> [] | i -> [ ("integrator", integrator_to_json i) ])
  )

let simulation_config_of_json j =
  { t_start        = as_float  (member "t_start"        j);
    t_end          = as_float  (member "t_end"          j);
    time_semantics = as_string (member "time_semantics" j);
    dt             = (match member_opt "dt"       j with Some `Null | None -> None | Some v -> Some (as_float v));
    rng_seed       = (match member_opt "rng_seed" j with Some `Null | None -> None | Some v -> Some (as_int   v));
    integrator     = (match member_opt "integrator" j with Some `Null | None -> Rk4 | Some v -> integrator_of_json v);
  }

(* ── Presets ─────────────────────────────────────────────────────────────── *)

let preset_to_json (p : preset) : Yojson.Safe.t =
  obj (
    [ ("name",    str p.preset_name);
      ("label",   str p.preset_label);
      ("params",  obj (List.map (fun (k, v) -> (k, flt v)) p.preset_params));
      ("enable",  arr (List.map str p.preset_enable));
      ("disable", arr (List.map str p.preset_disable));
      ("t_end",   match p.preset_t_end with None -> null | Some v -> flt v); ]
    @ (if p.preset_scale = [] then []
       else [("scale", obj (List.map (fun (k, v) -> (k, flt v)) p.preset_scale))])
    @ (if p.preset_compose = [] then []
       else [("compose", arr (List.map str p.preset_compose))])
  )

let preset_of_json j =
  { preset_name    = as_string (member "name"  j);
    preset_label   = as_string (member "label" j);
    preset_params  = List.map (fun (k, v) -> (k, as_float v)) (as_assoc (member "params" j));
    preset_enable  = (match member_opt "enable"  j with Some (`List xs) -> List.map as_string xs | _ -> []);
    preset_disable = (match member_opt "disable" j with Some (`List xs) -> List.map as_string xs | _ -> []);
    preset_scale   = (match member_opt "scale"   j with
                      | Some (`Assoc kvs) -> List.map (fun (k, v) -> (k, as_float v)) kvs
                      | _ -> []);
    preset_compose = (match member_opt "compose" j with Some (`List xs) -> List.map as_string xs | _ -> []);
    preset_t_end   = (match member_opt "t_end" j with Some `Null | None -> None | Some v -> Some (as_float v));
  }

(* ── Model structure ─────────────────────────────────────────────────────── *)

let dimension_to_json (d : dimension) : Yojson.Safe.t =
  obj [("name", str d.dim_name); ("values", arr (List.map str d.dim_values))]

let dimension_of_json j = {
  dim_name   = j |> member "name"   |> as_string;
  dim_values = j |> member "values" |> as_list |> List.map as_string;
}

let model_structure_to_json (ms : model_structure) : Yojson.Safe.t =
  obj [
    ("dimensions",               arr (List.map dimension_to_json ms.dimensions));
    ("compartment_dims",         obj (List.map (fun (k, vs) -> (k, arr (List.map str vs))) ms.compartment_dims));
    ("base_compartments",        arr (List.map str ms.base_compartments));
    ("transmission_transitions", arr (List.map str ms.transmission_transitions));
    ("infectious_compartments",  arr (List.map str ms.infectious_compartments));
  ]

let model_structure_of_json j = {
  dimensions = j |> member "dimensions" |> as_list |> List.map dimension_of_json;
  compartment_dims = j |> member "compartment_dims" |> as_assoc
    |> List.map (fun (k, v) -> (k, v |> as_list |> List.map as_string));
  base_compartments = j |> member "base_compartments" |> as_list |> List.map as_string;
  transmission_transitions = j |> member "transmission_transitions" |> as_list |> List.map as_string;
  infectious_compartments  = j |> member "infectious_compartments"  |> as_list |> List.map as_string;
}

(* ── Top-level model ─────────────────────────────────────────────────────── *)

let binding_to_json (b : binding) : Yojson.Safe.t =
  obj [("name", str b.bname); ("expr", expr_to_json b.bexpr)]

let binding_of_json (j : Yojson.Safe.t) : binding =
  { bname = as_string (member "name" j); bexpr = expr_of_json (member "expr" j) }

let model_to_json (m : model) : Yojson.Safe.t =
  obj ([
    ("name",               str m.name);
    ("version",            str m.version);
    ("time_unit",          str m.time_unit);
    ("description",        match m.description with None -> null | Some s -> str s);
  ] @ (match m.origin with None -> [] | Some s -> [("origin", str s)])
    @ (match m.origin_rata_die with None -> [] | Some n -> [("origin_rata_die", `Int n)]) @ [
    ("compartments",       arr (List.map compartment_to_json m.compartments));
    ("transitions",        arr (List.map transition_to_json m.transitions));
    ("ode_equations",      arr (List.map ode_equation_to_json m.ode_equations));
    ("time_functions",     arr (List.map time_function_to_json m.time_functions));
    ("tables",             arr (List.map table_to_json m.tables));
    ("interventions",      arr (List.map intervention_to_json m.interventions));
    ("observations",       arr (List.map observation_model_to_json m.observations));
    ("parameters",         arr (List.map parameter_to_json m.parameters));
    ("initial_conditions", initial_conditions_to_json m.initial_conditions);
    ("output",             output_config_to_json m.output);
    ("simulation",         simulation_config_to_json m.simulation);
    ("scenarios",          arr (List.map preset_to_json m.presets));
    ("model_structure",    match m.model_structure with None -> null | Some ms -> model_structure_to_json ms);
  ] @ (match m.balance with
       | None -> []
       | Some bs -> [("balance", obj [
           ("target", str bs.balance_target);
           ("expr",   expr_to_json bs.balance_expr);
         ])])
    @ (match m.identity_tracked_compartments with
       | [] -> []
       | cs -> [("identity_tracked_compartments", arr (List.map str cs))])
    @ (match m.bindings with
       | [] -> []
       | bs -> [("bindings", arr (List.map binding_to_json bs))])
    @ (match m.per_eval_bindings with
       | [] -> []
       | bs -> [("per_eval_bindings", arr (List.map binding_to_json bs))])
    @ (match m.quantities with
       | [] -> []
       | qs -> [("quantities", arr (List.map quantity_to_json qs))])
    @ (match m.contrasts with
       | [] -> []
       | cs -> [("contrasts", arr (List.map contrast_to_json cs))])
  )

let model_of_json (j : Yojson.Safe.t) : model =
  { name               = as_string (member "name"               j);
    version            = as_string (member "version"            j);
    time_unit          = (match member_opt "time_unit" j with Some (`String s) -> s | _ -> "days");
    description        = (match member_opt "description" j with Some `Null | None -> None | Some s -> Some (as_string s));
    origin             = (match member_opt "origin" j with Some (`String s) -> Some s | _ -> None);
    origin_rata_die    = (match member_opt "origin_rata_die" j with Some (`Int n) -> Some n | _ -> None);
    compartments       = List.map compartment_of_json      (as_list (member "compartments"   j));
    transitions        = List.map transition_of_json       (as_list (member "transitions"    j));
    ode_equations      = List.map ode_equation_of_json     (as_list (member "ode_equations"  j));
    time_functions     = List.map time_function_of_json    (as_list (member "time_functions" j));
    tables             = List.map table_of_json            (as_list (member "tables"         j));
    interventions      = List.map intervention_of_json     (as_list (member "interventions"  j));
    observations       = List.map observation_model_of_json (as_list (member "observations"  j));
    parameters         = List.map parameter_of_json        (as_list (member "parameters"     j));
    bindings           = (match member_opt "bindings" j with
                          | Some (`List v) -> List.map binding_of_json v | _ -> []);
    per_eval_bindings  = (match member_opt "per_eval_bindings" j with
                          | Some (`List v) -> List.map binding_of_json v | _ -> []);
    initial_conditions = initial_conditions_of_json (member "initial_conditions" j);
    output             = output_config_of_json     (member "output"     j);
    simulation         = simulation_config_of_json (member "simulation" j);
    presets            = (match member_opt "scenarios" j with
      | Some (`List v) -> List.map preset_of_json v
      | _ -> []);
    model_structure    = (match member_opt "model_structure" j with
      | None -> None
      | Some v -> opt_null model_structure_of_json v);
    balance            = (match member_opt "balance" j with
      | None -> None
      | Some v -> Some {
          balance_target = member "target" v |> as_string;
          balance_expr   = member "expr"   v |> expr_of_json;
        });
    identity_tracked_compartments =
      (match member_opt "identity_tracked_compartments" j with
       | None | Some `Null -> []
       | Some v -> List.map as_string (as_list v));
    (* The doc dictionary lives at the envelope level, not the model body;
       `model_of_envelope_json` reads it and overrides this default. *)
    doc_index          = empty_doc_index;
    quantities         = (match member_opt "quantities" j with
                          | Some (`List v) -> List.map quantity_of_json v | _ -> []);
    contrasts          = (match member_opt "contrasts" j with
                          | Some (`List v) -> List.map contrast_of_json v | _ -> []);
  }

(* gh#audit-C8. IR schema version baked at build time from `ir/VERSION`
   via the dune rule in this directory's `dune` file. Single source of
   truth: bumping `ir/VERSION` updates both OCaml (this constant) and
   Rust (envelope.rs's `include_str!`) on the next build. *)
let ir_version = Ir_version_generated.value

(* gh#audit-C8. Producer marker emitted in the IR envelope. Rust's
   validate.rs checks the marker; if present, can skip OCaml-mirrored
   structural checks (audit H14). *)
let validated_by = "ocaml-compiler-v" ^ ir_version

(* gh#audit-C8. Wrap the model in an envelope: { ir_version,
   validated_by, model: <existing> }. The Rust deserializer
   (rust/crates/ir/src/lib.rs) requires the wrapper and rejects
   mismatched ir_version with IrError::VersionMismatch. *)
let envelope_to_json (m : model) : Yojson.Safe.t =
  obj ([
    ("ir_version",   str ir_version);
    ("validated_by", str validated_by);
    ("model",        model_to_json m);
  ] @ (if doc_index_is_empty m.doc_index then []
       else [("docs", doc_index_to_json m.doc_index)]))

(* gh#audit-C8. Parse an envelope and verify the version handshake.
   On mismatch returns Error with a hint pointing the user at the
   right rebuild target. *)
let model_of_envelope_json (j : Yojson.Safe.t) : (model, string) result =
  match member_opt "ir_version" j with
  | Some (`String v) when v = ir_version ->
      (match member_opt "model" j with
       | Some mj ->
         (try
            let m = model_of_json mj in
            let doc_index = match member_opt "docs" j with
              | Some dj -> doc_index_of_json dj
              | None    -> empty_doc_index
            in
            Ok { m with doc_index }
          with DeserError msg -> Error msg)
       | None -> Error "IR envelope missing `model` field")
  | Some (`String v) ->
      Error (Printf.sprintf
        "IR version mismatch: this OCaml compiler emits %s, found %s in JSON. \
         Rebuild OCaml side (`make build-ocaml`) and re-emit any persisted IR JSON."
        ir_version v)
  | _ ->
      Error (Printf.sprintf
        "IR JSON missing `ir_version` field (expected wrapped envelope: \
         {ir_version, validated_by, model}). Re-emit IR with `make update-golden`.")

(* ── IR JSON output ────────────────────────────────────────────────────────
   The canonical on-disk format is COMPACT JSON with one element per line for
   the model's top-level arrays (compartments, transitions, parameters, …).
   Pretty-printing the IR was ~97% of compile time and ~80% of IR bytes on
   large models (docs/dev/notes/2026-05-30-compiler-profiling.md); compact
   removes both costs, while the per-element newlines keep golden diffs
   reviewable (a changed transition is a one-line diff). Both forms render the
   SAME `envelope_to_json m` value — they differ only in whitespace, which every
   JSON parser ignores — so they encode identical content. `~pretty:true`
   selects the indented human view (also available via `camdlc --pretty` and
   `camdlc inspect`). The canonical/pretty equivalence is pinned by
   canonical_equiv_test in test_ir_roundtrip.ml. *)

(* Render `envelope_to_json m` compactly, except that each element of the model
   object's non-empty array fields goes on its own line. [out_str] writes raw
   punctuation/whitespace; [out_json] renders a JSON value compactly via Yojson
   (to a channel or buffer — never an intermediate string, so peak memory stays
   at the AST, not AST + a multi-GB output string). The structure is read from
   the canonical AST — we iterate its fields rather than re-listing them — so no
   field can be silently dropped or reordered if the schema grows, and leaf
   bytes come from Yojson, so there is no token-level divergence from pretty. *)
let write_canonical
    ~(out_str : string -> unit) ~(out_json : Yojson.Safe.t -> unit)
    (m : model) : unit =
  let key k = out_json (`String k); out_str ":" in
  let write_list_per_line xs =
    out_str "[";
    List.iteri (fun i x -> if i > 0 then out_str ","; out_str "\n"; out_json x) xs;
    out_str "\n]"
  in
  let write_field (k, v) =
    key k;
    (match v with
     | `List (_ :: _ as xs) -> write_list_per_line xs
     | other                -> out_json other)
  in
  (match envelope_to_json m with
   | `Assoc efields ->
     out_str "{";
     List.iteri (fun i (k, v) ->
       if i > 0 then out_str ",";
       (match k, v with
        | "model", `Assoc mfields ->
          key k; out_str "{";
          List.iteri (fun j kv -> if j > 0 then out_str ","; write_field kv) mfields;
          out_str "}"
        | _ -> write_field (k, v))
     ) efields;
     out_str "}"
   | other -> out_json other)

let model_to_channel ?(pretty = false) (oc : out_channel) (m : model) : unit =
  if pretty then Yojson.Safe.pretty_to_channel oc (envelope_to_json m)
  else write_canonical ~out_str:(output_string oc)
         ~out_json:(Yojson.Safe.to_channel oc) m

let model_to_string ?(pretty = false) (m : model) : string =
  if pretty then Yojson.Safe.pretty_to_string (envelope_to_json m)
  else begin
    let b = Buffer.create 65536 in
    write_canonical ~out_str:(Buffer.add_string b)
      ~out_json:(Yojson.Safe.to_buffer b) m;
    Buffer.contents b
  end

let model_of_string (s : string) : (model, string) result =
  match Yojson.Safe.from_string s with
  | exception exn -> Error (Printexc.to_string exn)
  | j -> model_of_envelope_json j

let model_of_file (path : string) : (model, string) result =
  match Yojson.Safe.from_file path with
  | exception exn -> Error (Printexc.to_string exn)
  | j -> model_of_envelope_json j
