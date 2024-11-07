open Iree_bindings
open Dsl

let f = fn (List_type [Tensor_type ([], F32)]) (fun [x] -> Dsl.((x * x) + x))

let f' = Backpropagate.diff (VarCons Nil) f

let f' = Ir.compile f'

let () =
  print_endline f' ;
  Compile.compile f' "out.vmfb"
