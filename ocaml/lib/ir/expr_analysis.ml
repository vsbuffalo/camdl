(** The most dynamic thing an IR expression depends on, from a literal up to
    the projected value of an observation.

    {!dep_of_expr} classifies an {!Ir.expr} into one {!dep}, and
    {!model_binding_deps} classifies every binding of a model, each from
    the classes of the bindings it refers to. The classes form a chain,

    {v Const < Data < Param < Time < State < Projected v}

    and the class of an expression is the greatest class of its leaves
    ({!join}): [beta * S] is [State]. Its one user is the cost report of
    [camdlc inspect], which labels each binding with {!dep_name}.

    A class answers "how far does this expression vary", not "what does it
    read". Because the join keeps only the greatest class, [beta * S] is
    [State] although it reads the parameter [beta], so the class cannot say
    whether an expression is free of parameters; that is
    [Validate.references_param]. The differentiator and the runtime classify
    expressions for their own purposes with their own functions, and do not
    use this module.

    {!dep_of_expr} names every {!Ir.expr} constructor and has no side
    effects. It raises [Failure] on an {!Ir.PerEvalRef}, which exists only
    after the last compiler pass. *)

(* NB: we do not `open Ir` — this module's [dep] constructors `Const` and
   `Param` would shadow `Ir.Const` / `Ir.Param`. IR expression
   constructors are qualified explicitly in the match below. *)

(** How far an expression can vary: the class of the most dynamic quantity
    it reads. Constructors are listed from least to most dynamic. *)
type dep =
  | Const      (** A literal, a model constant or a unit literal: fixed when the model is compiled. *)
  | Data       (** A table cell or an observation's data: fixed once a run's data is bound. A table lookup is [Data] whatever its cells hold, so a lookup into a table of parameters is [Data], not [Param]. *)
  | Param      (** A model parameter: fixed for one simulation, different between them. *)
  | Time       (** Simulation time, the step [dt], or a forcing. *)
  | State      (** A compartment: changes as the system advances. *)
  | Projected  (** The projected value inside a likelihood argument. *)

(* Position in the chain. join is max-by-rank, which is a valid
   semilattice join precisely because the order is a total chain. *)
let rank = function
  | Const     -> 0
  | Data      -> 1
  | Param     -> 2
  | Time      -> 3
  | State     -> 4
  | Projected -> 5

let join a b = if rank a >= rank b then a else b

let join_list ds = List.fold_left join Const ds

(* Lowercase one-word label, for reports / diagnostics. *)
let dep_name = function
  | Const     -> "const"
  | Data      -> "data"
  | Param     -> "param"
  | Time      -> "time"
  | State     -> "state"
  | Projected -> "projected"

(* Classify an expression. [binding_dep name] resolves a [BindingRef] to
   the (already-computed) class of the named binding; callers that have no
   binding environment can pass [fun _ -> Const] (a BindingRef then floors
   to Const, which is only safe when the model has no bindings — use
   [model_binding_deps] otherwise). *)
let dep_of_expr ~binding_dep (e : Ir.expr) : dep =
  let rec go : Ir.expr -> dep = function
    | Ir.Const _ -> Const
    | Ir.Param _ -> Param
    (* A model constant is a compile-time literal by another name. *)
    | Ir.ConstRef _ -> Const
    | Ir.Pop _ | Ir.PopSum _ -> State
    | Ir.Time | Ir.Dt | Ir.TimeFunc _ | Ir.ForcingBasis _ -> Time
    (* A table lookup is at least Data (its compile-time cells); its index
       expressions may pull it more-dynamic (e.g. a state-indexed lookup). *)
    | Ir.TableLookup (_, idxs) -> join Data (join_list (List.map go idxs))
    | Ir.BindingRef name -> binding_dep name
    | Ir.PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"
    | Ir.Projected -> Projected
    (* A per-observation aux column is external data supplied at load, not a
       function of simulator state — classify as Data (like a constant-indexed
       table cell). It is differentiated to 0 (a data constant) in autodiff. *)
    | Ir.ObsColumnRef _ -> Data
    (* An observation anchor is external data supplied by the run (the
       observed window), and it is substituted for a `Const` before evaluation,
       so it classifies as Data — the same class as a table cell. It appears
       only in a forcing `breakpoints` list, never in a rate. *)
    | Ir.ObsAnchor _ -> Data
    | Ir.BinOp b -> join (go b.left) (go b.right)
    | Ir.UnOp u  -> go u.arg
    | Ir.Cond c  -> join (go c.pred) (join (go c.then_) (go c.else_))
    | Ir.Reduce terms -> join_list (List.map go terms)
    | Ir.UncheckedDim u -> go u.inner
    | Ir.UnitLiteral _ -> Const
  in
  go e

(* Compute each binding's dep in topological order. `m.bindings` is
   topo-ordered (a BindingRef only references an earlier binding), so a
   single forward pass suffices: each binding resolves its own
   BindingRefs against the deps already accumulated. An unknown name
   (shouldn't happen on a valid model) floors to Const. *)
let model_binding_deps (m : Ir.model) : string -> dep =
  let tbl : (string, dep) Hashtbl.t = Hashtbl.create 16 in
  let lookup name = match Hashtbl.find_opt tbl name with Some d -> d | None -> Const in
  List.iter
    (fun (b : Ir.binding) ->
      let d = dep_of_expr ~binding_dep:lookup b.bexpr in
      Hashtbl.replace tbl b.bname d)
    m.bindings;
  lookup
