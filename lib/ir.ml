type id = int

let current_id = ref 0

let new_id () =
  let id = !current_id in
  current_id := id + 1 ;
  id

let tag x = (new_id (), x)

type f32 = F32

type i1 = I1

type i64 = I64

type _ tensor_element_type =
  | F32 : f32 tensor_element_type
  | I1 : i1 tensor_element_type
  | I64 : i64 tensor_element_type

type shape = int list

type 'a value_type = Tensor_type of shape * 'a tensor_element_type

let tensor_element_type_to_stable_hlo :
    type a. a tensor_element_type -> Stable_hlo.tensor_element_type = function
  | F32 ->
      Stable_hlo.F32
  | I1 ->
      Stable_hlo.I1
  | I64 ->
      Stable_hlo.I64

let value_type_to_stable_hlo : type a. a value_type -> Stable_hlo.value_type =
  function
  | Tensor_type (shape, tensor_element_type) ->
      Stable_hlo.Tensor_type
        (shape, tensor_element_type_to_stable_hlo tensor_element_type)

type 'a tagged = id * 'a

type comparison_direction = Eq | Ne | Ge | Gt | Le | Lt

let attribute_of_comparison_direction direction =
  ( "comparison_direction"
  , "#stablehlo<comparison_direction "
    ^ ( match direction with
      | Eq ->
          "EQ"
      | Ne ->
          "NE"
      | Ge ->
          "GE"
      | Gt ->
          "GT"
      | Le ->
          "LE"
      | Lt ->
          "LT" )
    ^ ">" )

type distribution = Uniform | Normal

module rec Var : sig
  type 'a t = 'a expr tagged

  and _ expr =
    | Add : 'a t * 'a t -> 'a expr
    | Subtract : 'a t * 'a t -> 'a expr
    | Multiply : 'a t * 'a t -> 'a expr
    | Abs : 'a t -> 'a expr
    | Argument : 'a value_type -> 'a expr
    | Compare : 'a t * comparison_direction * 'a t -> i1 expr
    | Constant : 'a value_type * string -> 'a expr
    | Random : 'a value_type * f32 t * f32 t * i64 t * distribution -> 'a expr
    | Output : 'a value_type * ('b, 'c) Call.t -> 'a expr
end = struct
  type 'a t = 'a expr tagged

  and _ expr =
    | Add : 'a t * 'a t -> 'a expr
    | Subtract : 'a t * 'a t -> 'a expr
    | Multiply : 'a t * 'a t -> 'a expr
    | Abs : 'a t -> 'a expr
    | Argument : 'a value_type -> 'a expr
    | Compare : 'a t * comparison_direction * 'a t -> i1 expr
    | Constant : 'a value_type * string -> 'a expr
    | Random : 'a value_type * f32 t * f32 t * i64 t * distribution -> 'a expr
    | Output : 'a value_type * ('b, 'c) Call.t -> 'a expr
end

and Call : sig
  type ('a, 'b) t =
    { func: ('a, 'b) Func.t
    ; args: ((unit * 'b) VarList.t * 'a) VarList.t
    ; output_ids: int list
    ; id: id }
end = struct
  type ('a, 'b) t =
    { func: ('a, 'b) Func.t
    ; args: ((unit * 'b) VarList.t * 'a) VarList.t
    ; output_ids: int list
    ; id: id }
end

and VarList :
  (Hlist.S with type 'a u = 'a Var.t and type 'a v = 'a Var.expr tagged) =
Hlist.Make (struct
  type 'a t = 'a Var.t

  type 'a tag = 'a Var.expr tagged
end)

and ValueTypeList :
  (Hlist.S with type 'a u = 'a value_type and type 'a v = 'a Var.expr tagged) =
Hlist.Make (struct
  type 'a t = 'a value_type

  type 'a tag = 'a Var.expr tagged
end)

and Func : sig
  type ('a, 'b) t =
    { inputs: ((unit * 'b) VarList.t * 'a) ValueTypeList.t
    ; parameter_names: string list
    ; outputs: (unit * 'b) VarList.t
    ; name: string }
end = struct
  type ('a, 'b) t =
    { inputs: ((unit * 'b) VarList.t * 'a) ValueTypeList.t
    ; parameter_names: string list
    ; outputs: (unit * 'b) VarList.t
    ; name: string }
end

let rec value_type_of_var : type a. a Var.t -> a value_type = function
  | _, Add (lhs, _) ->
      value_type_of_var lhs
  | _, Subtract (lhs, _) ->
      value_type_of_var lhs
  | _, Multiply (lhs, _) ->
      value_type_of_var lhs
  | _, Abs var ->
      value_type_of_var var
  | _, Argument value_type ->
      value_type
  | _, Compare (a, _, _) ->
      let (Tensor_type (shape, _)) = value_type_of_var a in
      Tensor_type (shape, I1)
  | _, Constant (value_type, _) ->
      value_type
  | _, Random (value_type, _, _, _, _) ->
      value_type
  | _, Output (value_type, _) ->
      value_type

let var_to_annotated_value var =
  (string_of_int @@ fst var, value_type_to_stable_hlo @@ value_type_of_var var)

module AnnotatedValueSet = Set.Make (struct
  type t = string * Stable_hlo.value_type

  let compare = Stdlib.compare
end)

module IntMap = Map.Make (Int)

let vars_to_ops vars =
  let rec aux :
      type a. Stable_hlo.op IntMap.t -> a Var.t -> Stable_hlo.op IntMap.t =
   fun cache (id, expr) ->
    if IntMap.mem id cache then cache
    else
      let output = var_to_annotated_value (id, expr) in
      match expr with
      | Add (lhs, rhs) ->
          let cache = aux cache lhs in
          let cache = aux cache rhs in
          let op =
            Stable_hlo.
              { inputs= [var_to_annotated_value lhs; var_to_annotated_value rhs]
              ; outputs= [output]
              ; name= "stablehlo.add"
              ; attributes= []
              ; call= false }
          in
          IntMap.add id op cache
      | Subtract (lhs, rhs) ->
          let cache = aux cache lhs in
          let cache = aux cache rhs in
          let op =
            Stable_hlo.
              { inputs= [var_to_annotated_value lhs; var_to_annotated_value rhs]
              ; outputs= [output]
              ; name= "stablehlo.subtract"
              ; attributes= []
              ; call= false }
          in
          IntMap.add id op cache
      | Multiply (lhs, rhs) ->
          let cache = aux cache lhs in
          let cache = aux cache rhs in
          let op =
            Stable_hlo.
              { inputs= [var_to_annotated_value lhs; var_to_annotated_value rhs]
              ; outputs= [output]
              ; name= "stablehlo.multiply"
              ; attributes= []
              ; call= false }
          in
          IntMap.add id op cache
      | Abs var ->
          let cache = aux cache var in
          let op =
            Stable_hlo.
              { inputs= [var_to_annotated_value var]
              ; outputs= [output]
              ; name= "stablehlo.abs"
              ; attributes= []
              ; call= false }
          in
          IntMap.add id op cache
      | Argument _ ->
          cache
      | Compare (lhs, direction, rhs) ->
          let cache = aux cache lhs in
          let cache = aux cache rhs in
          let op =
            Stable_hlo.
              { inputs= [var_to_annotated_value lhs; var_to_annotated_value rhs]
              ; outputs= [output]
              ; name= "stablehlo.compare"
              ; attributes= [attribute_of_comparison_direction direction]
              ; call= false }
          in
          IntMap.add id op cache
      | Constant (_, repr) ->
          let op =
            Stable_hlo.
              { inputs= []
              ; outputs= [output]
              ; name= "stablehlo.constant"
              ; attributes= [("value", repr)]
              ; call= false }
          in
          IntMap.add id op cache
      | Random (_, a, b, shape, distribution) ->
          let cache = aux cache a in
          let cache = aux cache b in
          let cache = aux cache shape in
          let op =
            Stable_hlo.
              { inputs=
                  [ var_to_annotated_value a
                  ; var_to_annotated_value b
                  ; var_to_annotated_value shape ]
              ; outputs= [output]
              ; name= "stablehlo.rng"
              ; attributes=
                  [ ( "rng_distribution"
                    , "#stablehlo<rng_distribution "
                      ^ ( match distribution with
                        | Uniform ->
                            "UNIFORM"
                        | Normal ->
                            "NORMAL" )
                      ^ ">" ) ]
              ; call= false }
          in
          IntMap.add id op cache
      | Output (_, call) ->
          if IntMap.mem call.id cache then cache
          else
            let cache = VarList.fold_left {f= aux} cache call.args in
            let inputs =
              VarList.map_to_list {f= var_to_annotated_value} call.args
            in
            let outputs =
              VarList.map2_to_list
                {f= (fun (_, expr) id -> var_to_annotated_value (id, expr))}
                call.func.outputs call.output_ids
            in
            let op =
              Stable_hlo.
                { inputs
                ; outputs
                ; name= call.func.name
                ; attributes= []
                ; call= true }
            in
            IntMap.add call.id op cache
  in
  VarList.fold_left {f= aux} IntMap.empty vars
  |> IntMap.bindings |> List.map snd

let annotated_values_to_return_op values =
  Stable_hlo.
    { inputs= values
    ; outputs= []
    ; name= "func.return"
    ; attributes= []
    ; call= false }

let rec apply :
    type a b.
    (a VarList.t * b) ValueTypeList.t -> b -> a VarList.t * string list =
 fun inputs body ->
  match inputs with
  | [] ->
      (body, [])
  | input :: inputs ->
      let id = new_id () in
      let input = (id, Var.Argument input) in
      let output, parameters = apply inputs (body input) in
      (output, string_of_int id :: parameters)

let create_func :
    type a b. ((unit * b) VarList.t * a) ValueTypeList.t -> a -> (a, b) Func.t =
 fun inputs body ->
  let outputs, parameter_names = apply inputs body in
  {inputs; parameter_names; outputs; name= "fn" ^ string_of_int (new_id ())}

let func_to_stable_hlo (func : ('a, 'b) Func.t) =
  let ops = vars_to_ops func.outputs in
  let inputs =
    ValueTypeList.map_to_list {f= value_type_to_stable_hlo} func.inputs
  in
  let inputs = List.combine func.parameter_names inputs in
  let outputs = VarList.map_to_list {f= var_to_annotated_value} func.outputs in
  let return_ops = annotated_values_to_return_op outputs in
  let outputs = List.map snd outputs in
  Stable_hlo.{id= func.name; inputs; outputs; body= ops @ [return_ops]}

let call_func :
    type a b.
       (a, b) Func.t
    -> ((unit * b) VarList.t * a) VarList.t
    -> (unit * b) VarList.t =
 fun func args ->
  let output_ids = VarList.map_to_list {f= (fun _ -> new_id ())} func.outputs in
  let call = Call.{func; args; output_ids; id= new_id ()} in
  VarList.map2
    {f= (fun var id -> (id, Var.Output (value_type_of_var var, call)))}
    func.outputs output_ids

module StringMap = Map.Make (String)

let compile entry =
  let rec all_funcs :
      type a. string StringMap.t -> a Var.t -> string StringMap.t =
   fun cache var ->
    match var with
    | _, Var.Add (a, b) ->
        all_funcs (all_funcs cache a) b
    | _, Subtract (a, b) ->
        all_funcs (all_funcs cache a) b
    | _, Multiply (a, b) ->
        all_funcs (all_funcs cache a) b
    | _, Abs a ->
        all_funcs cache a
    | _, Argument _ ->
        cache
    | _, Compare (a, _, b) ->
        all_funcs (all_funcs cache a) b
    | _, Constant _ ->
        cache
    | _, Random (_, a, b, c, _) ->
        all_funcs (all_funcs (all_funcs cache a) b) c
    | _, Output (_, call) ->
        let cache = VarList.fold_left {f= all_funcs} cache call.args in
        let func = call.func in
        if StringMap.mem func.name cache then cache
        else
          let str = func_to_stable_hlo func |> Stable_hlo.func_to_string in
          StringMap.add func.name str cache
  in
  let main = func_to_stable_hlo entry |> Stable_hlo.func_to_string in
  let cache = StringMap.add entry.Func.name main StringMap.empty in
  let funcs = VarList.fold_left {f= all_funcs} cache entry.Func.outputs in
  StringMap.bindings funcs |> List.map snd |> String.concat "\n"
