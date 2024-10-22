type id = int

let current_id = ref 0

let new_id () =
  let id = !current_id in
  current_id := id + 1 ;
  id

type 'a tagged = id * 'a

type input = unit

type var = expr tagged

and expr = Absf of var | Argument of input

let ( let* ) untagged fn = fn (new_id (), untagged)

let lift fn (_, value) = fn value

let expr_to_value_type = function
  | Absf _ ->
      Stable_hlo.(Tensor_type ([], F32))
  | Argument _ ->
      Stable_hlo.(Tensor_type ([], F32))

let var_to_annotated_value (id, expr) =
  (string_of_int id, expr_to_value_type expr)

module AnnotatedValueSet = Set.Make (struct
  type t = string * Stable_hlo.value_type

  let compare = Stdlib.compare
end)

let rec get_inputs var =
  let aux (id, expr) =
    match expr with
    | Absf v ->
        get_inputs v
    | Argument _ ->
        AnnotatedValueSet.singleton (var_to_annotated_value (id, expr))
  in
  aux var

module IntMap = Map.Make (Int)

let vars_to_ops vars =
  let rec aux cache (id, expr) =
    if IntMap.mem id cache then cache
    else
      match expr with
      | Absf var ->
          let cache = aux cache var in
          let op =
            Stable_hlo.
              { inputs=
                  [(string_of_int @@ fst var, expr_to_value_type @@ snd var)]
              ; outputs= [(string_of_int id, expr_to_value_type expr)]
              ; name= "math.absf" }
          in
          IntMap.add id op cache
      | Argument _ ->
          cache
  in
  List.fold_left aux IntMap.empty vars |> IntMap.bindings |> List.map snd

let annotated_values_to_return_op values =
  Stable_hlo.{inputs= values; outputs= []; name= "func.return"}

let vars_to_func vars =
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
