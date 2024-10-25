type id = int

let current_id = ref 0

let new_id () =
  let id = !current_id in
  current_id := id + 1 ;
  id

type f32 = F32

type i1 = I1

type _ tensor_element_type =
  | F32 : f32 tensor_element_type
  | I1 : i1 tensor_element_type

type shape = int list

type 'a value_type = Tensor_type of shape * 'a tensor_element_type

let tensor_element_type_to_stable_hlo :
    type a. a tensor_element_type -> Stable_hlo.tensor_element_type = function
  | F32 ->
      Stable_hlo.F32
  | I1 ->
      Stable_hlo.I1

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

type 'a var = 'a expr tagged

and _ expr =
  | Absf : 'a var -> 'a expr
  | Argument : 'a value_type -> 'a expr
  | Compare : 'a var * comparison_direction * 'a var -> i1 expr

let rec value_type_of_var : type a. a var -> a value_type = function
  | _, Absf var ->
      value_type_of_var var
  | _, Argument value_type ->
      value_type
  | _, Compare (a, _, _) ->
      let (Tensor_type (shape, _)) = value_type_of_var a in
      Tensor_type (shape, I1)

module ValueTypeList = Hlist.Make (struct
  type 'a t = 'a value_type

  type 'a tag = 'a expr tagged
end)

module VarList = Hlist.Make (struct
  type 'a t = 'a var

  type 'a tag = 'a
end)

let ( let* ) untagged fn = fn (new_id (), untagged)

let lift fn (_, value) = fn value

let var_to_annotated_value var =
  (string_of_int @@ fst var, value_type_to_stable_hlo @@ value_type_of_var var)

module AnnotatedValueSet = Set.Make (struct
  type t = string * Stable_hlo.value_type

  let compare = Stdlib.compare
end)

let rec get_inputs : type a. a var -> AnnotatedValueSet.t =
 fun (id, expr) ->
  match expr with
  | Absf v ->
      get_inputs v
  | Argument _ ->
      AnnotatedValueSet.singleton (var_to_annotated_value (id, expr))
  | Compare (lhs, _, rhs) ->
      AnnotatedValueSet.union (get_inputs lhs) (get_inputs rhs)

module IntMap = Map.Make (Int)

let vars_to_ops vars =
  let rec aux :
      type a. Stable_hlo.op IntMap.t -> a var -> Stable_hlo.op IntMap.t =
   fun cache (id, expr) ->
    if IntMap.mem id cache then cache
    else
      let output = var_to_annotated_value (id, expr) in
      match expr with
      | Absf var ->
          let cache = aux cache var in
          let op =
            Stable_hlo.
              { inputs= [var_to_annotated_value var]
              ; outputs= [output]
              ; name= "math.absf"
              ; attributes= [] }
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
              ; attributes= [attribute_of_comparison_direction direction] }
          in
          IntMap.add id op cache
  in
  VarList.fold_left {f= aux} IntMap.empty vars
  |> IntMap.bindings |> List.map snd

let annotated_values_to_return_op values =
  Stable_hlo.{inputs= values; outputs= []; name= "func.return"; attributes= []}

type ('a, 'b) func =
  { inputs: 'a ValueTypeList.t
  ; parameter_names: string list
  ; outputs: 'b VarList.t }

let rec apply :
    type a b.
    (a VarList.t * b) ValueTypeList.t -> b -> a VarList.t * string list =
 fun inputs body ->
  match inputs with
  | [] ->
      (body, [])
  | input :: inputs ->
      let id = new_id () in
      let input = (id, Argument input) in
      let output, parameters = apply inputs (body input) in
      (output, string_of_int id :: parameters)

let create_func :
    type a b.
    (a VarList.t * b) ValueTypeList.t -> b -> (a VarList.t * b, a) func =
 fun inputs body ->
  let outputs, parameter_names = apply inputs body in
  {inputs; parameter_names; outputs}

let func_to_stable_hlo (func : ('a, 'b) func) =
  let ops = vars_to_ops func.outputs in
  let inputs = ValueTypeList.map {f= value_type_to_stable_hlo} func.inputs in
  let inputs = List.combine func.parameter_names inputs in
  let outputs = VarList.map {f= var_to_annotated_value} func.outputs in
  let return_ops = annotated_values_to_return_op outputs in
  let outputs = List.map snd outputs in
  Stable_hlo.
    { id= "func" ^ string_of_int @@ new_id ()
    ; inputs
    ; outputs
    ; body= ops @ [return_ops] }
