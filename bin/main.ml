open Iree_bindings
open Dsl

let add =
  fn [Tensor_type ([], F32); Tensor_type ([], F32)] (fun x y -> [x + y; x + x])

let add' = Backpropagate.(differentiate_func (Const @-> Var @-> return) add)

let call_add' =
  fn
    [Tensor_type ([], F32); Tensor_type ([], F32)]
    (fun x y ->
      let [a; b; c] = Ir.call_func add' [x; y] in
      [a + b + c] )

let main =
  fn
    [Tensor_type ([], F32)]
    (fun x ->
      let [x; y] = Ir.call_func add [x; x] in
      [x + y] )

let main_code = Ir.compile main

let () =
  print_endline main_code ;
  Compile.compile main_code "out.vmfb"
