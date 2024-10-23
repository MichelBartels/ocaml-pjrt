type id = int

let current_id = ref 0

let new_id () =
  let id = !current_id in
  current_id := id + 1 ;
  id

type f32

type i1

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
  | Absf : f32 var -> f32 expr
  | Argument : int list -> 'a expr
  | Compare : 'a var * comparison_direction * 'a var -> i1 expr

type any_var = AnyVar : 'a var -> any_var

type _ var_list =
  | [] : unit var_list
  | ( :: ) : 'a var * 'b var_list -> ('a * 'b) var_list

let rec var_list_to_any_var_list : type a. a var_list -> any_var list = function
  | [] ->
      []
  | var :: rest ->
      AnyVar var :: var_list_to_any_var_list rest

let arg : int expr = Argument []

let ( let* ) untagged fn = fn (new_id (), untagged)

let lift fn (_, value) = fn value

let rec expr_to_value_type : type a. a expr -> Stable_hlo.value_type = function
  | Absf (_, expr) ->
      expr_to_value_type expr
  | Argument shape ->
      Stable_hlo.(Tensor_type (shape, F32))
  | Compare (_, _, _) ->
      Stable_hlo.(Tensor_type ([], I1))

let var_to_annotated_value (AnyVar (id, expr)) =
  (string_of_int id, expr_to_value_type expr)

module AnnotatedValueSet = Set.Make (struct
  type t = string * Stable_hlo.value_type

  let compare = Stdlib.compare
end)

let rec get_inputs : any_var -> AnnotatedValueSet.t =
 fun (AnyVar (id, expr)) ->
  match expr with
  | Absf v ->
      get_inputs (AnyVar v)
  | Argument _ ->
      AnnotatedValueSet.singleton (var_to_annotated_value (AnyVar (id, expr)))
  | Compare (lhs, _, rhs) ->
      AnnotatedValueSet.union (get_inputs (AnyVar lhs))
        (get_inputs (AnyVar rhs))

module IntMap = Map.Make (Int)

let vars_to_ops : any_var list -> Stable_hlo.op list =
 fun vars ->
  let rec aux : Stable_hlo.op IntMap.t -> any_var -> Stable_hlo.op IntMap.t =
   fun cache (AnyVar (id, expr)) ->
    if IntMap.mem id cache then cache
    else
      match expr with
      | Absf var ->
          let cache = aux cache (AnyVar var) in
          let op =
            Stable_hlo.
              { inputs=
                  [(string_of_int @@ fst var, expr_to_value_type @@ snd var)]
              ; outputs= [(string_of_int id, expr_to_value_type expr)]
              ; name= "math.absf"
              ; attributes= [] }
          in
          IntMap.add id op cache
      | Argument _ ->
          cache
      | Compare (lhs, direction, rhs) ->
          let cache = aux cache (AnyVar lhs) in
          let cache = aux cache (AnyVar rhs) in
          let op =
            Stable_hlo.
              { inputs=
                  [ (string_of_int @@ fst lhs, expr_to_value_type @@ snd lhs)
                  ; (string_of_int @@ fst rhs, expr_to_value_type @@ snd rhs) ]
              ; outputs= [(string_of_int id, expr_to_value_type expr)]
              ; name= "stablehlo.compare"
              ; attributes= [attribute_of_comparison_direction direction] }
          in
          IntMap.add id op cache
  in
  List.fold_left aux IntMap.empty vars |> IntMap.bindings |> List.map snd

let annotated_values_to_return_op values =
  Stable_hlo.{inputs= values; outputs= []; name= "func.return"; attributes= []}

let vars_to_func vars =
  let vars = var_list_to_any_var_list vars in
  let ops = vars_to_ops vars in
  let inputs =
    List.fold_left
      (fun acc var -> AnnotatedValueSet.union acc (get_inputs var))
      AnnotatedValueSet.empty vars
    |> AnnotatedValueSet.elements
  in
  let outputs = List.map var_to_annotated_value vars in
  let return_ops = annotated_values_to_return_op outputs in
  let outputs = List.map snd outputs in
  Stable_hlo.
    { id= "func" ^ string_of_int @@ new_id ()
    ; inputs
    ; outputs
    ; body= ops @ [return_ops] }
