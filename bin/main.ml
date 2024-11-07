open Iree_bindings
open Dsl

let f' = Backpropagate.diff Var (fun x -> [(x * x) + x])

let g x =
  let [[[grad]]; [value]] = f' x in
  grad + value

let g = fn (Tensor_type ([], F32)) g

let g = Ir.compile g

let () =
  print_endline g ;
  Compile.compile g "out.vmfb"
