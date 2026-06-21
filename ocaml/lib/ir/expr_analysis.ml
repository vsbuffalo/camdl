(* Dependency classification for IR expressions.

   A single source of truth for "what does this expression depend on?",
   replacing the two ad-hoc one-bit classifiers that exist today (each a
   projection of this lattice, computed independently per language):
   - OCaml `autodiff.ml` treats `BindingRef` as param-free (d/dp = 0);
   - Rust `resolved_expr.rs::references_state` treats `BindingRef` as
     state-derived.

   The dependency classes form a join-semilattice. `Const` is the least
   element (a literal depends on nothing); `join` returns the more-dynamic
   of two classes. The chain is the declaration order below:

     Const  <  Data  <  Param  <  Time  <  State  <  Projected

   so `join` is "the one further along the chain". `references_state` is a
   clean projection of this lattice:
     references_state(e)  ≡  dep e ⊒ State   (State or Projected)

   NOTE: the lattice does NOT capture "references a param". `join` keeps only
   the MOST-dynamic class, so `beta * S` classifies as `State` (the `Param` is
   absorbed) even though it references `beta`. "Param-free" — the hoist/autodiff
   invariant that a `BindingRef` differentiates to 0 (E512) — is therefore a
   separate STRUCTURAL check (`Validate.references_param`), NOT a projection of
   this lattice. Do not use `dep e ≠ Param` to mean param-free; it is not.

   Pure and total: every constructor of `Ir.expr` is classified, no
   exceptions, no side effects. *)

(* NB: we do NOT `open Ir` — this module's [dep] constructors `Const` and
   `Param` would shadow `Ir.Const` / `Ir.Param`. IR expression
   constructors are qualified explicitly in the match below. *)

type dep =
  | Const      (* literal: depends on nothing *)
  | Data       (* compile-time table data (constant-indexed lookups) *)
  | Param      (* model parameter (estimable / runtime-supplied) *)
  | Time       (* simulation time, dt, or a time function of them *)
  | State      (* compartment populations (varies as the system advances) *)
  | Projected  (* projection output in a likelihood (state-derived) *)

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
    | Ir.Pop _ | Ir.PopSum _ -> State
    | Ir.Time | Ir.Dt | Ir.TimeFunc _ -> Time
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
    | Ir.BinOp b -> join (go b.left) (go b.right)
    | Ir.UnOp u  -> go u.arg
    | Ir.Cond c  -> join (go c.pred) (join (go c.then_) (go c.else_))
    | Ir.Reduce terms -> join_list (List.map go terms)
    | Ir.UncheckedDim u -> go u.inner
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
