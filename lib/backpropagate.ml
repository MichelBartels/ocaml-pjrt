(* type (_, _, _, _) input_list = *)
(*   | Nil : ('a, 'a, 'b, 'b) input_list *)
(*   | ConstCons : ('a, 'b, 'c, 'd) input_list -> ('a, 'e -> 'b, 'c, 'd) input_list *)
(*   | VarCons : *)
(*       ('a, 'b, 'c, 'd) input_list *)
(*       -> ('a, 'e -> 'b, 'c, 'e -> 'd) input_list *)

(* let return = Nil *)

(* type (_, _, _) input_type = *)
(*   | Const : ('a, 'b, 'b) input_type *)
(*   | Var : ('a, 'a -> 'c, 'c) input_type *)

(* let ( @-> ) : *)
(*     type a b c d e f. *)
(*        (a, b, c) input_type *)
(*     -> (f, d, e, c) input_list *)
(*     -> (f, a -> d, e, b) input_list = *)
(*  fun t l -> match t with Const -> ConstCons l | Var -> VarCons l *)

(* let rec extract_vars_from_value_type_list : *)
(*     type a b c d. *)
(*        (a, b, c, d) input_list *)
(*     -> (a * b) Ir.ValueTypeList.t *)
(*     -> (c * d) Ir.ValueTypeList.t = *)
(*  fun l l' -> *)
(*   match (l, l') with *)
(*   | Nil, _ -> *)
(*       [] *)
(*   | ConstCons l, _ :: l' -> *)
(*       extract_vars_from_value_type_list l l' *)
(*   | VarCons l, v :: l' -> *)
(*       v :: extract_vars_from_value_type_list l l' *)
(*   | _ -> *)
(*       failwith "should be impossible" *)

(* module StringMap = Map.Make (String) *)

(* let rec differentiate_var : type a. int -> a Ir.Var.t -> a Ir.Var.t = *)
(*  fun var_id (id, v) -> *)
(*   if id = var_id then Dsl.ones_like (id, v) *)
(*   else *)
(*     Dsl.( *)
(*       match v with *)
(*       | Add (v1, v2) -> *)
(*           differentiate_var var_id v1 + differentiate_var var_id v2 *)
(*       | Subtract (v1, v2) -> *)
(*           differentiate_var var_id v1 - differentiate_var var_id v2 *)
(*       | Multiply (v1, v2) -> *)
(*           (v1 * differentiate_var var_id v2) + (v2 * differentiate_var var_id v1) *)
(*       | Abs v -> *)
(*           abs (differentiate_var var_id v) *)
(*       | Argument _ -> *)
(*           zeros_like (id, v) *)
(*       | Compare (a, dir, b) -> *)
(*           compare dir (differentiate_var var_id a) (differentiate_var var_id b) *)
(*       | Constant (value_type, _) -> ( *)
(*         match value_type with *)
(*         | Ir.Tensor_type (shape, F32) -> *)
(*             full (F32 0.0) shape *)
(*         | Ir.Tensor_type (_, I1) -> *)
(*             failwith "differentiation of I1 not supported" *)
(*         | Ir.Tensor_type (_, I64) -> *)
(*             failwith "differentiation of I64 not supported" ) *)
(*       | Random _ -> *)
(*           ones_like (id, v) *)
(*       | Output _ -> *)
(*           failwith "todo" ) *)

(* let differentiate_func : *)
(*     type a b c d. *)
(*     (a, b, c, d) input_list -> (b, c, a) Ir.Func.t -> (b, d, a) Ir.Func.t = *)
(*  fun l f -> *)
(*   let var_types = extract_vars_from_value_type_list l f.Ir.Func.inputs in *)
(*   let open Hlist.Map (Ir.ValueTypeList) (Ir.VarList) in *)
(*   let vars = map {f= (fun _ -> failwith "")} var_types in *)
(*   let outputs = Ir.VarList.append vars f.Ir.Func.outputs in *)
(*   Ir.Func.{f with outputs} *)

(* (input_type, old_output_type, new_output_type) *)
type (_, _, _) input_list =
  | Nil : (unit, 'a, 'a Ir.Var.t -> unit) input_list
  | ConstCons :
      ('a, 'b, 'c) input_list
      -> ('d Ir.tensor Ir.Var.t -> 'a, 'b, 'c) input_list
  | VarCons :
      ('a, 'b, 'c) input_list
      -> ('d Ir.tensor Ir.Var.t -> 'a, 'b, 'b Ir.Var.t -> 'c) input_list

type (_, _, _) input =
  | Var : ('a Ir.tensor Ir.Var.t, 'b, 'b Ir.Var.t -> unit) input
  | Const : ('a Ir.tensor Ir.Var.t, 'a, unit) input
  | [] : (unit Ir.VarList.t Ir.Var.t, 'a, unit Ir.VarList.t Ir.Var.t) input
  | ( :: ) :
      ('a, 'b, 'c) input
      * ('d Ir.VarList.t Ir.Var.t, 'b, 'e Ir.VarList.t Ir.Var.t) input
      -> ( ('a -> 'd) Ir.VarList.t Ir.Var.t
         , 'a
         , 'c -> 'e Ir.VarList.t Ir.Var.t )
         input

let diff' :
    type a b c.
       (a, b, c) input
    -> (a -> b Ir.Var.t)
    -> a Ir.VarList.t Ir.Var.t
    -> (c -> b -> unit) Ir.VarList.t Ir.Var.t =
 fun _ _ _ -> failwith "todo"

let diff :
    type a b c.
       (a, b, c) input_list
    -> (a Ir.VarList.t Ir.Var.t -> b Ir.Var.t)
    -> a Ir.VarList.t Ir.Var.t
    -> c Ir.VarList.t Ir.Var.t =
 fun l f inputs ->
  let rec backprop : type a. a Ir.Var.t -> int -> a Ir.Var.t =
    Dsl.(
      fun v x ->
        match v with
        | Ir.Var.Add (v1, v2) ->
            backprop v1 x + backprop v2 x
        | Subtract (v1, v2) ->
            backprop v1 x - backprop v2 x
        | Multiply (v1, v2) ->
            (v1 * backprop v2 x) + (v2 * backprop v1 x)
        | Abs v ->
            abs (backprop v x)
        | Argument _ ->
            zeros_like v
        | Compare (a, dir, b) ->
            compare dir (backprop a x) (backprop b x)
        | Constant _ ->
            zeros_like v
        | Random _ ->
            zeros_like v
        | [] ->
            []
        | y :: ys ->
            backprop y x :: backprop ys x
        | DiffVar (id, _) ->
            if Int.equal id x then Dsl.ones_like v else Dsl.zeros_like v
        | DiffConst _ ->
            Dsl.zeros_like v )
  in
  let rec wrap_inputs :
      type a b c.
         (a, b, c) input_list
      -> a Ir.VarList.t Ir.Var.t
      -> a Ir.VarList.t Ir.Var.t * Ir.id list =
   fun l1 l2 ->
    match (l1, l2) with
    | Nil, [] ->
        ([], [])
    | ConstCons l, x :: xs ->
        let inputs, ids = wrap_inputs l xs in
        (DiffConst x :: inputs, ids)
    | VarCons l, x :: xs ->
        let inputs, ids = wrap_inputs l xs in
        let id = Ir.new_id () in
        (DiffVar (id, x) :: inputs, id :: ids)
  in
  let rec iter_vars :
      type a b c.
         (a, b, c) input_list
      -> b Ir.Var.t
      -> Ir.id list
      -> c Ir.VarList.t Ir.Var.t =
   fun l outputs names ->
    match (l, names) with
    | Nil, [] ->
        [outputs]
    | ConstCons l, _ :: names ->
        iter_vars l outputs names
    | VarCons l, name :: names ->
        backprop outputs name :: iter_vars l outputs names
    | _ ->
        failwith "should be impossible"
  in
  let inputs, ids = wrap_inputs l inputs in
  let outputs = f inputs in
  iter_vars l outputs ids
