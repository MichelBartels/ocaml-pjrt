open Iree_bindings
open Dsl

let _ = Backpropagate.diff' Var (fun x -> [x + x])

let f' = Backpropagate.diff (VarCons Nil) (fun [x] -> [x + x])

let f' = fn (List_type [Tensor_type ([], F32)]) f'

let f' = Ir.compile f'

let () =
  print_endline f' ;
  Compile.compile f' "out.vmfb"
