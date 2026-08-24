(* Dependency order for `init { }` (gh#733).

   An initial-condition spec may read another compartment's initial value
   (`init { I = I0   S = N0 - I }`), so the entries form a directed graph over
   compartment references and have to be evaluated in topological order:
   dependencies first, each entry against the partially built state. A
   reference cycle has no evaluation order and is rejected.

   One implementation, two consumers:
   - [Validate] rejects a cycle (E515) at the contract boundary;
   - [Compiler] walks the DAG to inline each referenced compartment's own
     initial expression before differentiating the ODE forward-sensitivity
     seed (`ic_grad`), so the seed differentiates the value the runtime
     actually builds.

   The Rust runtime (`sim/src/compiled_model.rs`) sorts the same graph the same
   way at `CompiledModel::new`; this module is the OCaml half of that
   agreement. *)

module SM = Map.Make (String)

(* Compartments an initial-condition expression reads.

   [BindingRef] is followed into the model's hoisted bindings: `let N = S+I+R`
   used in an init RHS is a real dependency on S, I and R, because the runtime
   evaluates the binding body against whatever state exists at that moment.
   Treating it as a leaf would order the entry wrongly and read zeros. Bindings
   are emitted in dependency order and are acyclic; [seen] is a belt-and-braces
   guard so a drifted or hand-written IR cannot spin here. *)
let deps ~(bindings : Ir.binding list) (e : Ir.expr) : string list =
  let body_of =
    List.fold_left
      (fun m (b : Ir.binding) -> SM.add b.bname b.bexpr m)
      SM.empty bindings
  in
  let acc = ref [] in
  let seen_bindings = Hashtbl.create 8 in
  let add c = if not (List.mem c !acc) then acc := c :: !acc in
  let rec go (e : Ir.expr) =
    match e with
    | Ir.Const _ | Ir.Param _ | Ir.Time | Ir.Dt | Ir.TimeFunc _ | Ir.Projected
    | Ir.ObsColumnRef _ | Ir.ObsAnchor _ -> ()
    | Ir.Pop c -> add c
    | Ir.PopSum cs -> List.iter add cs
    | Ir.BinOp b -> go b.left; go b.right
    | Ir.UnOp u -> go u.arg
    | Ir.Cond c -> go c.pred; go c.then_; go c.else_
    | Ir.TableLookup (_, idxs) -> List.iter go idxs
    | Ir.UncheckedDim u -> go u.inner
    | Ir.Reduce terms -> List.iter go terms
    | Ir.PerEvalRef _ ->
      (* LICM (`licm.ml`) rewrites transition rates and observation arguments;
         it never touches `initial_conditions`, and it runs after both
         consumers of this module. A `PerEvalRef` here means the IR drifted. *)
      failwith "PerEvalRef in an initial condition (gh#272 compiler invariant)"
    | Ir.BindingRef n ->
      if not (Hashtbl.mem seen_bindings n) then begin
        Hashtbl.replace seen_bindings n ();
        match SM.find_opt n body_of with Some b -> go b | None -> ()
      end
  in
  go e;
  List.rev !acc

(* Topologically sort the init entries.

   [Ok names] is a permutation of the entry names with every dependency before
   its dependant; ties are broken by declaration order, so the result is
   deterministic. [Error cycle] names the compartments on one reference cycle,
   in the order they close it (`A -> B -> A` reports `["A"; "B"]`).

   A referenced compartment with no init entry is NOT an edge: it starts at 0
   (the default) and constrains nothing. *)
let topo (ic : Ir.initial_conditions) ~(bindings : Ir.binding list)
  : (string list, string list) result =
  (* EVERY expression the spec evaluates is an edge source: a law's arguments
     are evaluated against the partially built state exactly as a deterministic
     RHS is, so `I ~ binomial(n = N0 - R, p = q)` depends on `R`. *)
  let entry =
    List.fold_left (fun m (k, s) -> SM.add k (Ir.init_spec_exprs s) m) SM.empty ic in
  (* absent = unvisited, `Grey = on the current DFS path, `Black = finished *)
  let state : (string, [ `Grey | `Black ]) Hashtbl.t = Hashtbl.create 32 in
  let out = ref [] in
  let cycle = ref None in
  let rec visit path name =
    match Hashtbl.find_opt state name with
    | Some `Black -> ()
    | Some `Grey ->
      if !cycle = None then begin
        (* [path] is the current DFS path, innermost first. The cycle is the
           prefix up to and including the re-entered node, re-oriented so it
           reads in dependency order. *)
        let rec take = function
          | [] -> []
          | x :: rest -> if x = name then [x] else x :: take rest
        in
        cycle := Some (List.rev (take path))
      end
    | _ ->
      (match SM.find_opt name entry with
       | None -> ()   (* no init entry: starts at 0, imposes no order *)
       | Some es ->
         Hashtbl.replace state name `Grey;
         List.iter
           (fun e ->
              List.iter
                (fun d -> if SM.mem d entry then visit (name :: path) d)
                (deps ~bindings e))
           es;
         Hashtbl.replace state name `Black;
         out := name :: !out)
  in
  List.iter (fun (k, _) -> visit [] k) ic;
  match !cycle with
  | Some c -> Error c
  | None -> Ok (List.rev !out)

(* Each init expression closed over the rest of the block: every [Pop c] is
   replaced by c's own (already closed) initial expression, every [PopSum] by
   the sum of those, and every [BindingRef] by its closed body. The result is a
   function of parameters and constants alone, in declaration order.

   This is what the IC gradient must differentiate. `ic_grad` is the ODE
   forward-sensitivity seed S(t_start) = ∂(initial_state)/∂θ, and after gh#733
   `initial_state` reads other compartments' initial values — so differentiating
   the RAW expression, where [Autodiff.differentiate] sends [Pop] and
   [BindingRef] to 0, reports a derivative of a value the runtime does not
   compute. Concretely `init { A = A0   B = A0 - A }` has ∂B/∂A0 = 0 (B is
   identically 0), but the raw expression differentiates to 1.

   A referenced compartment with no init entry closes to [Const 0.0], matching
   the runtime, where an unseeded compartment starts at 0.

   A LAW entry closes over its MEAN expression. `ic_grad` seeds the ODE forward
   sensitivity, and the ODE path is deterministic: it starts every compartment
   at [Ir.init_spec_mean_expr]. Closing over anything else would differentiate a
   value that path never computes. *)
let closed (ic : Ir.initial_conditions) ~(bindings : Ir.binding list)
  : (string * Ir.expr) list =
  match topo ic ~bindings with
  | Error _ ->
    (* [Validate] rejects a cycle (E515) and the compile pipeline short-circuits
       on validate errors before this pass runs. *)
    failwith "init dependency cycle reached the IC-gradient pass (gh#733)"
  | Ok order ->
    (* The MEAN expression, for a law: the ODE forward-sensitivity seed
       differentiates the value the deterministic path starts from, and that
       path takes each law at its mean. *)
    let entry =
      List.fold_left
        (fun m (k, s) -> SM.add k (Ir.init_spec_mean_expr s) m) SM.empty ic in
    let body_of =
      List.fold_left
        (fun m (b : Ir.binding) -> SM.add b.bname b.bexpr m)
        SM.empty bindings
    in
    let built = ref SM.empty in
    let rec subst (e : Ir.expr) : Ir.expr =
      match e with
      | Ir.Const _ | Ir.Param _ | Ir.Time | Ir.Dt | Ir.TimeFunc _ | Ir.Projected
      | Ir.ObsColumnRef _ | Ir.ObsAnchor _ -> e
      | Ir.Pop c -> (match SM.find_opt c !built with Some ce -> ce | None -> Ir.Const 0.0)
      | Ir.PopSum cs ->
        (match cs with
         | [] -> Ir.Const 0.0
         | _ -> Ir.Reduce (List.map (fun c -> subst (Ir.Pop c)) cs))
      | Ir.BinOp b -> Ir.BinOp { b with left = subst b.left; right = subst b.right }
      | Ir.UnOp u -> Ir.UnOp { u with arg = subst u.arg }
      | Ir.Cond c ->
        Ir.Cond { pred = subst c.pred; then_ = subst c.then_; else_ = subst c.else_ }
      | Ir.TableLookup (t, idxs) -> Ir.TableLookup (t, List.map subst idxs)
      | Ir.UncheckedDim u -> Ir.UncheckedDim { u with inner = subst u.inner }
      | Ir.Reduce terms -> Ir.Reduce (List.map subst terms)
      | Ir.PerEvalRef _ ->
        failwith "PerEvalRef in an initial condition (gh#272 compiler invariant)"
      | Ir.BindingRef n ->
        (match SM.find_opt n body_of with Some b -> subst b | None -> e)
    in
    List.iter
      (fun name -> built := SM.add name (subst (SM.find name entry)) !built)
      order;
    List.map (fun (k, _) -> (k, SM.find k !built)) ic
