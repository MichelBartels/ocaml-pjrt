type (_, _, _, _) input_list =
  | Nil : ('a, 'a, 'b, 'b) input_list
  | ConstCons : ('a, 'b, 'c, 'd) input_list -> ('a, 'e -> 'b, 'c, 'd) input_list
  | VarCons :
      ('a, 'b, 'c, 'd) input_list
      -> ('a, 'e -> 'b, 'c, 'e -> 'd) input_list

let return = Nil

type (_, _, _) input_type =
  | Const : ('a, 'b, 'b) input_type
  | Var : ('a, 'a -> 'c, 'c) input_type

let ( @-> ) :
    type a b c d e f.
       (a, b, c) input_type
    -> (f, d, e, c) input_list
    -> (f, a -> d, e, b) input_list =
 fun t l -> match t with Const -> ConstCons l | Var -> VarCons l

let rec extract_vars_from_value_type_list :
    type a b c d.
       (a, b, c, d) input_list
    -> (a * b) Ir.ValueTypeList.t
    -> (c * d) Ir.ValueTypeList.t =
 fun l l' ->
  match (l, l') with
  | Nil, _ ->
      []
  | ConstCons l, _ :: l' ->
      extract_vars_from_value_type_list l l'
  | VarCons l, v :: l' ->
      v :: extract_vars_from_value_type_list l l'
  | _ ->
      failwith "should be impossible"

module StringMap = Map.Make (String)

let rec differentiate_var : type a. int -> a Ir.Var.t -> a Ir.Var.t =
 fun var_id (id, v) ->
  if id = var_id then Dsl.ones_like (id, v)
  else
    Dsl.(
      match v with
      | Add (v1, v2) ->
          differentiate_var var_id v1 + differentiate_var var_id v2
      | Subtract (v1, v2) ->
          differentiate_var var_id v1 - differentiate_var var_id v2
      | Multiply (v1, v2) ->
          (v1 * differentiate_var var_id v2) + (v2 * differentiate_var var_id v1)
      | Abs v ->
          abs (differentiate_var var_id v)
      | Argument _ ->
          zeros_like (id, v)
      | Compare (a, dir, b) ->
          compare dir (differentiate_var var_id a) (differentiate_var var_id b)
      | Constant (value_type, _) -> (
        match value_type with
        | Ir.Tensor_type (shape, F32) ->
            full (F32 0.0) shape
        | Ir.Tensor_type (_, I1) ->
            failwith "differentiation of I1 not supported"
        | Ir.Tensor_type (_, I64) ->
            failwith "differentiation of I64 not supported" )
      | Random _ ->
          ones_like (id, v)
      | Output _ ->
          failwith "todo" )

let differentiate_func :
    type a b c d.
    (a, b, c, d) input_list -> (b, c, a) Ir.Func.t -> (b, d, a) Ir.Func.t =
 fun l f ->
  let var_types = extract_vars_from_value_type_list l f.Ir.Func.inputs in
  let open Hlist.Map (Ir.ValueTypeList) (Ir.VarList) in
  let vars = map {f= (fun _ -> failwith "")} var_types in
  let outputs = Ir.VarList.append vars f.Ir.Func.outputs in
  Ir.Func.{f with outputs}
