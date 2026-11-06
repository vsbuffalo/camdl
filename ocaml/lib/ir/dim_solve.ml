(* Exact integer solver for dimension equations. See dim_solve.mli. *)

type var = int

type stage = Declared | Structure | Obligation

type origin = { site : int; stage : stage; what : string }

type eqn = { terms : (var * int) list; rhs : int array; origin : origin }

type conflict_kind = Contradiction | Not_integral

type conflict = {
  culprit : eqn;
  kind    : conflict_kind;
  context : origin list;
}

type relation = { coeffs : (var * int) list; equals : int array }

exception Overflow of string

(* ── Checked arithmetic ────────────────────────────────────────────────── *)

let overflow op a b =
  raise (Overflow (Printf.sprintf "dimension solver: %d %s %d overflows" a op b))

let ( +! ) a b =
  let s = a + b in
  if (a >= 0) = (b >= 0) && (s >= 0) <> (a >= 0) then overflow "+" a b else s

let neg a = if a = min_int then overflow "-" 0 a else - a

let ( -! ) a b = a +! neg b

let ( *! ) a b =
  if a = 0 || b = 0 then 0
  else begin
    let p = a * b in
    if (a = -1 && b = min_int) || (b = -1 && a = min_int) || p / b <> a
    then overflow "*" a b
    else p
  end

let checked_add = ( +! )
let checked_mul = ( *! )

let vec_axpy c x y = Array.mapi (fun i yi -> yi -! (c *! x.(i))) y   (* y - c·x *)

(* ── Sparse rows ───────────────────────────────────────────────────────── *)

module IM = Map.Make (Int)

let add_coef v c m =
  match IM.find_opt v m with
  | None -> if c = 0 then m else IM.add v c m
  | Some c0 -> let s = c0 +! c in if s = 0 then IM.remove v m else IM.add v s m

let terms_of_list l = List.fold_left (fun m (v, c) -> add_coef v c m) IM.empty l

(* A unit row: x_pivot + Σ terms = rhs. [terms] never mentions a pivot of any
   row (the accepted rows are kept fully reduced against one another). *)
type row = {
  mutable terms : int IM.t;
  mutable rhs   : int array;
  mutable ctx   : origin list;
}

let context_cap = 12

let merge_ctx a b =
  let rec go acc n = function
    | [] -> List.rev acc
    | _ when n >= context_cap -> List.rev acc
    | o :: rest -> if List.mem o acc then go acc n rest else go (o :: acc) (n + 1) rest
  in
  go [] 0 (a @ b)

type state = {
  nbases   : int;
  rows     : (var, row) Hashtbl.t;
  occ      : (var, (var, unit) Hashtbl.t) Hashtbl.t;   (* var → pivots of rows mentioning it *)
  mutable residual : (eqn * origin list) list;          (* reversed *)
  mutable conflicts : conflict list;                    (* reversed *)
}

let occ_of st v =
  match Hashtbl.find_opt st.occ v with
  | Some h -> h
  | None -> let h = Hashtbl.create 4 in Hashtbl.replace st.occ v h; h

(* Substitute every pivot variable out of [terms], using the accepted rows. *)
let reduce st terms rhs ctx =
  IM.fold (fun v c (terms, rhs, ctx) ->
    match Hashtbl.find_opt st.rows v with
    | None -> (terms, rhs, ctx)
    | Some r ->
      (* terms - c·(x_v + r.terms) *)
      let terms = IM.remove v terms in
      let terms = IM.fold (fun j cj m -> add_coef j (neg (c *! cj)) m) r.terms terms in
      (terms, vec_axpy c r.rhs rhs, merge_ctx ctx r.ctx))
    terms (terms, rhs, ctx)

let is_zero v = Array.for_all (( = ) 0) v

let add_unit_row st pivot terms rhs ctx =
  let s = IM.find pivot terms in
  let terms = if s = 1 then terms else IM.map neg terms in
  let rhs = if s = 1 then rhs else Array.map neg rhs in
  let terms = IM.remove pivot terms in
  (* Eliminate the new pivot from every row that mentions it. *)
  let users = occ_of st pivot in
  Hashtbl.iter (fun p () ->
    let r = Hashtbl.find st.rows p in
    let c = IM.find pivot r.terms in
    r.terms <- IM.remove pivot r.terms;
    IM.iter (fun j cj ->
      let before = IM.mem j r.terms in
      r.terms <- add_coef j (neg (c *! cj)) r.terms;
      let after = IM.mem j r.terms in
      if after && not before then Hashtbl.replace (occ_of st j) p ()
      else if before && not after then Hashtbl.remove (occ_of st j) p) terms;
    r.rhs <- vec_axpy c rhs r.rhs;
    r.ctx <- merge_ctx r.ctx ctx) users;
  Hashtbl.reset users;
  Hashtbl.replace st.rows pivot { terms; rhs; ctx };
  IM.iter (fun j _ -> Hashtbl.replace (occ_of st j) pivot ()) terms

let add_eqn st (e : eqn) =
  if Array.length e.rhs <> st.nbases then
    invalid_arg "Dim_solve.solve: rhs length differs from nbases";
  let terms, rhs, ctx = reduce st (terms_of_list e.terms) (Array.copy e.rhs) [e.origin] in
  if IM.is_empty terms then begin
    if not (is_zero rhs) then
      st.conflicts <- { culprit = e; kind = Contradiction;
                        context = ctx } :: st.conflicts
  end else
    let unit_pivot =
      IM.fold (fun v c best -> if abs c = 1 then Some v else best) terms None
      (* IM.fold visits in increasing order, so this is the newest variable *)
    in
    match unit_pivot with
    | Some p -> add_unit_row st p terms rhs ctx
    | None -> st.residual <- (e, ctx) :: st.residual

(* ── Column Hermite form ───────────────────────────────────────────────── *)

(* Extended gcd: (g, x, y) with x·a + y·b = g ≥ 0. *)
let rec ext_gcd a b =
  if b = 0 then (if a < 0 then (neg a, -1, 0) else (a, 1, 0))
  else
    let q = a / b and r = a mod b in
    let g, x, y = ext_gcd b r in
    (g, y, x -! (q *! y))

(* Reduce the k×m matrix [a] in place to column echelon form by unimodular
   column operations, applied to [u] (m columns) too. Returns the pivot column
   of each row ([-1] for a row without one) and the rank. *)
let column_echelon (a : int array array) (u : int array array) m =
  let k = Array.length a in
  let col_op i j f =
    (* columns (i, j) ← f(col_i, col_j), elementwise, in a and u *)
    let apply mat = Array.iter (fun row ->
      let x, y = f row.(i) row.(j) in row.(i) <- x; row.(j) <- y) mat in
    apply a; apply u
  in
  let pivot_of_row = Array.make k (-1) in
  let col = ref 0 in
  for r = 0 to k - 1 do
    if !col < m then begin
      let c0 = !col in
      for c = c0 + 1 to m - 1 do
        let b = a.(r).(c) in
        if b <> 0 then begin
          let av = a.(r).(c0) in
          if av = 0 then col_op c0 c (fun x y -> (y, x))
          else begin
            let g, x, y = ext_gcd av b in
            let p = av / g and q = b / g in
            (* [col0; col] ← [x·col0 + y·col; −q·col0 + p·col], det = x·p + y·q = 1 *)
            col_op c0 c (fun s t -> ((x *! s) +! (y *! t), (neg q *! s) +! (p *! t)))
          end
        end
      done;
      if a.(r).(c0) <> 0 then begin
        if a.(r).(c0) < 0 then col_op c0 c0 (fun s _ -> (neg s, neg s));
        pivot_of_row.(r) <- c0;
        incr col
      end
    end
  done;
  (pivot_of_row, !col)

(* Solve the echelon system for one base. [`Ok y] with free columns at 0,
   [`Not_divisible], or [`Inconsistent] (a row without a pivot not satisfied). *)
let forward_solve a pivot_of_row m b =
  let y = Array.make m 0 in
  let k = Array.length a in
  let rec go r =
    if r = k then `Ok y
    else begin
      let s = ref b.(r) in
      Array.iteri (fun c yc -> if c <> pivot_of_row.(r) then s := !s -! (a.(r).(c) *! yc)) y;
      match pivot_of_row.(r) with
      | -1 -> if !s = 0 then go (r + 1) else `Inconsistent
      | pc ->
        let d = a.(r).(pc) in
        if !s mod d <> 0 then `Not_divisible
        else (y.(pc) <- !s / d; go (r + 1))
    end
  in
  go 0

(* The residual system: rows of (terms, rhs) over the variables [vars]. *)
type echelon = {
  u       : int array array;     (* m×m, x = U·y *)
  rank    : int;
  y0      : int array array;     (* per base, the particular solution in y *)
}

let matrix_of rows cols =
  let index = Hashtbl.create 16 in
  Array.iteri (fun j v -> Hashtbl.replace index v j) cols;
  let m = Array.length cols in
  Array.of_list (List.map (fun (terms, _) ->
    let row = Array.make m 0 in
    IM.iter (fun v c -> row.(Hashtbl.find index v) <- c) terms; row) rows)

let identity m = Array.init m (fun i -> Array.init m (fun j -> if i = j then 1 else 0))

(* Feasibility of [rows] in integers, per base. *)
let analyse nbases rows =
  let cols =
    List.fold_left (fun s (terms, _) -> IM.fold (fun v _ s -> IM.add v () s) terms s)
      IM.empty rows
    |> IM.bindings |> List.map fst |> Array.of_list
  in
  let m = Array.length cols in
  let a = matrix_of rows cols in
  let u = identity m in
  let pivot_of_row, rank = column_echelon a u m in
  let rhs b = Array.of_list (List.map (fun (_, r) -> r.(b)) rows) in
  let results = List.init nbases (fun b -> forward_solve a pivot_of_row m (rhs b)) in
  (cols, u, rank, results)

let classify nbases rows =
  let _, _, _, results = analyse nbases rows in
  if List.exists (function `Inconsistent -> true | _ -> false) results then
    Some Contradiction
  else if List.exists (function `Not_divisible -> true | _ -> false) results then
    (* Echelon rows without a pivot were all satisfied, but a division was not
       exact. Rational solvability is decided by rank: [A | b] has the rank of
       A exactly when a rational solution exists. *)
    let rank_of rows =
      let _, _, rank, _ = analyse nbases rows in rank in
    let r = rank_of rows in
    let augmented b =
      List.map (fun (terms, rhs) -> (add_coef max_int rhs.(b) terms, rhs)) rows in
    if List.exists (fun b -> rank_of (augmented b) > r) (List.init nbases Fun.id)
    then Some Contradiction else Some Not_integral
  else None

(* ── The outcome ───────────────────────────────────────────────────────── *)

(* A variable's general solution: x0 + Σ coef·t over free parameters t, keyed
   [`Var v] for a variable no row constrains, [`Col j] for a free column of
   the residual's echelon form. *)
type affine = { x0 : int array; m : ((([ `Var of int | `Col of int ]) * int) list) }

type outcome = {
  st        : state;
  echelon   : echelon option;
  res_index : (var, int) Hashtbl.t;
  memo      : (var, affine) Hashtbl.t;
}

let solve ~nbases eqns =
  let st = { nbases; rows = Hashtbl.create 256; occ = Hashtbl.create 256;
             residual = []; conflicts = [] } in
  List.iter (add_eqn st) eqns;
  (* The residual rows, re-reduced against the final unit rows (some of their
     variables may have been pivoted later), accepted one at a time so that an
     infeasible one is reported and dropped like any other conflict. *)
  let accepted =
    List.fold_left (fun acc ((e : eqn), ctx) ->
      let terms, rhs, ctx = reduce st (terms_of_list e.terms) (Array.copy e.rhs) ctx in
      if IM.is_empty terms then begin
        if not (is_zero rhs) then
          st.conflicts <- { culprit = e; kind = Contradiction;
                            context = ctx } :: st.conflicts;
        acc
      end else
        let candidate = acc @ [ (terms, rhs) ] in
        match classify nbases candidate with
        | None -> candidate
        | Some kind ->
          st.conflicts <- { culprit = e; kind; 
                            context = ctx } :: st.conflicts;
          acc)
      [] (List.rev st.residual)
  in
  let echelon, res_index =
    let index = Hashtbl.create 16 in
    if accepted = [] then (None, index)
    else begin
      let cols, u, rank, results = analyse nbases accepted in
      Array.iteri (fun j v -> Hashtbl.replace index v j) cols;
      let y0 = Array.of_list (List.map (function
        | `Ok y -> y
        | `Not_divisible | `Inconsistent ->
          (* every accepted set was classified feasible *)
          assert false) results) in
      (Some { u; rank; y0 }, index)
    end
  in
  { st; echelon; res_index; memo = Hashtbl.create 256 }

let conflicts o = List.rev o.st.conflicts

let rec affine_of o v =
  match Hashtbl.find_opt o.memo v with
  | Some a -> a
  | None ->
    let nb = o.st.nbases in
    let a =
      match Hashtbl.find_opt o.st.rows v with
      | Some r ->
        (* x_v = rhs − Σ c_j·x_j, every x_j a non-pivot variable *)
        IM.fold (fun j c acc ->
          let aj = affine_of o j in
          { x0 = vec_axpy c aj.x0 acc.x0;
            m = List.fold_left (fun m (k, cj) ->
                  let prev = Option.value ~default:0 (List.assoc_opt k m) in
                  let s = prev -! (c *! cj) in
                  let m = List.remove_assoc k m in
                  if s = 0 then m else (k, s) :: m) acc.m aj.m })
          r.terms { x0 = Array.copy r.rhs; m = [] }
      | None ->
        (match o.echelon, Hashtbl.find_opt o.res_index v with
         | Some e, Some i ->
           let urow = e.u.(i) in
           let x0 = Array.init nb (fun b ->
             let s = ref 0 in
             Array.iteri (fun j uij -> s := !s +! (uij *! e.y0.(b).(j))) urow; !s) in
           let m = ref [] in
           Array.iteri (fun j uij ->
             if j >= e.rank && uij <> 0 then m := (`Col j, uij) :: !m) urow;
           { x0; m = !m }
         | _ -> { x0 = Array.make nb 0; m = [ (`Var v, 1) ] })
    in
    Hashtbl.replace o.memo v a;
    a

let value o v =
  let a = affine_of o v in
  if a.m = [] then Some a.x0 else None

let relations o vars =
  let vars = Array.of_list vars in
  let p = Array.length vars in
  if p = 0 then []
  else begin
    let affs = Array.map (affine_of o) vars in
    let keys =
      Array.fold_left (fun ks a ->
        List.fold_left (fun ks (k, _) -> if List.mem k ks then ks else k :: ks) ks a.m)
        [] affs |> List.rev |> Array.of_list in
    let f = Array.length keys in
    (* Mᵀ is f×p; its integer nullspace is the relations' coefficient basis. *)
    let mt = Array.init f (fun r -> Array.init p (fun c ->
      Option.value ~default:0 (List.assoc_opt keys.(r) affs.(c).m))) in
    let u = identity p in
    let _, rank = column_echelon mt u p in
    List.filter_map (fun j ->
      let coeffs = List.filter (fun (_, c) -> c <> 0)
          (List.init p (fun i -> (vars.(i), u.(i).(j)))) in
      if coeffs = [] then None
      else
        let equals = Array.init o.st.nbases (fun b ->
          List.fold_left (fun s (i, c) -> s +! (c *! affs.(i).x0.(b))) 0
            (List.init p (fun i -> (i, u.(i).(j))))) in
        Some { coeffs; equals })
      (List.init (p - rank) (fun d -> rank + d))
  end
